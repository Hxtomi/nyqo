"""N1 native LFM runtime, exact span masks and group-normalized text CE.

No evaluation data is read by training. No remote code or network model loading.
"""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                  HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path('/workspace/Nyqo')
B0=ROOT/'models/LiquidAI--LFM2.5-350M/9e6c6ccf47cd318696e137d381a7ded8fe4df09f'
DOMAIN=('source_use','instructions','reasoning_math','knowledge_uncertainty','dialogue_context','code')
PARTS=('final','reason','keep')
WEIGHTS={'final':1.,'reason':.25,'keep':1.}
SEED=221026

def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f: json.dump(obj,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def seed_all(seed=SEED):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False

def load_model(path=B0,training=False,attention='sdpa'):
    tok=AutoTokenizer.from_pretrained(str(B0),local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(str(path),local_files_only=True,trust_remote_code=False,
                                            dtype=torch.bfloat16,attn_implementation=attention)
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    vocab=model.get_input_embeddings().weight
    for p in model.parameters():
        p.requires_grad_(training and p is not vocab)
        if p.requires_grad: p.data=p.data.float()
    # Resume checkpoints carry true FP32 body. Restore those without BF16 rounding.
    if training and Path(path)!=B0:
        from safetensors.torch import load_file
        state=load_file(str(Path(path)/'model.safetensors'))
        model.load_state_dict(state,strict=False)
        del state
    model.to('cuda');model.train(training)
    assert sum(p.numel() for p in model.parameters())==354483968
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==(287375104 if training else 0)
    assert vocab.dtype==torch.bfloat16
    assert all(p.dtype==torch.float32 for p in model.parameters() if p.requires_grad)
    model.config.use_cache=not training
    return model,tok

def parse_answer(raw):
    """Only a leading structural block is reasoning; quoted/literal mentions stay literal."""
    body=raw.split('<|im_end|>',1)[0].strip()
    if not body.startswith('<think>'):
        return {'final':body,'reason':'','status':'plain' if body else 'missing_final'}
    tail=body[len('<think>'):]
    if '</think>' not in tail: return {'final':'','reason':tail,'status':'unclosed_reason'}
    reason,final=tail.split('</think>',1)
    if '<think>' in reason or '</think>' in reason:
        return {'final':'','reason':reason,'status':'nested_reason'}
    final=final.strip()
    # A second leading structural block is ambiguous. Literal mentions in prose/code are retained.
    if final.startswith(('<think>','</think>')):
        return {'final':'','reason':reason,'status':'repeated_reason'}
    return {'final':final,'reason':reason,'status':'reason_final' if final else 'missing_final'}

def encode(row,tok,branch='A',max_length=2048):
    assert row['language']=='en' and row['domain'] in DOMAIN and row['kind'] in ['new','keep']
    assert all(m['role'] in ['system','user','assistant'] for m in row['messages'])
    assert row['messages'][-1]['role']=='user'
    # History contains delivered final only, never a teacher trace or privileged answer.
    assert all(not any(k in m for k in ['reasoning','thinking','reasoning_content']) for m in row['messages'])
    reason=row.get('reason','') if branch=='B' and row['need_reason'] and row['kind']=='new' else ''
    if branch=='B' and row['need_reason'] and row['kind']=='new': assert reason.strip(),row['id']
    assert row['final'].strip()
    prefix=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True)
    answer={'role':'assistant','content':row['final']}
    if reason: answer['reasoning']=reason
    text=tok.apply_chat_template(row['messages']+[answer],tokenize=False,add_generation_prompt=False)
    reasoning='<think>'+reason+'</think>' if reason else ''
    expected=prefix+reasoning+row['final']+'<|im_end|>\n'
    assert text==expected,'native template mismatch'
    end_reason=len(prefix)+len(reasoning)
    end_final=end_reason+len(row['final'])+len('<|im_end|>')
    enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    ids=enc['input_ids']; offsets=enc['offset_mapping']
    assert len(ids)<=max_length,(row['id'],len(ids),max_length)
    masks={k:[0]*len(ids) for k in PARTS}
    spans={'reason':(len(prefix),end_reason),'keep' if row['kind']=='keep' else 'final':(end_reason,end_final)}
    for part,(start,end) in spans.items():
        for j,(a,b) in enumerate(offsets):
            if start<=a<b<=end: masks[part][j]=1
            elif a<end and b>start and start!=end:
                raise AssertionError(('token crosses supervised boundary',row['id'],part,j,(a,b),(start,end)))
    assert all(sum(masks[k][i] for k in PARTS)<=1 for i in range(len(ids)))
    assert masks['final'].count(1)+masks['keep'].count(1)>0
    supervised=[i for i in range(len(ids)) if any(masks[k][i] for k in PARTS)]
    assert ids[max(supervised)]==tok.eos_token_id==7
    assert min(supervised)>0
    return {'id':row['id'],'ids':ids,'masks':masks,'counts':{k:sum(masks[k]) for k in PARTS},'text':text}

def denominator(group): return {k:sum(x['counts'][k] for x in group) for k in PARTS}

def sample_loss(model,example,denoms,full_logits=False):
    ids=torch.tensor([example['ids']],device='cuda',dtype=torch.long)
    positions=torch.tensor([i for i in range(1,len(example['ids'])) if any(example['masks'][k][i] for k in PARTS)],device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out=model(input_ids=ids,use_cache=False,logits_to_keep=0 if full_logits else positions-1)
        logits=out.logits[0,positions-1] if full_logits else out.logits[0]
        ce=F.cross_entropy(logits.float(),ids[0,positions],reduction='none')
        loss=ce.sum()*0
        sums={}
        for part in PARTS:
            mask=torch.tensor(example['masks'][part],device='cuda',dtype=torch.bool)[positions]
            part_sum=ce[mask].sum()
            sums[part]=float(part_sum.detach())
            if denoms[part]: loss=loss+WEIGHTS[part]*part_sum/denoms[part]
    return loss,sums

def make_optimizer(model):
    return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-5,
                             betas=(.9,.95),eps=1e-8,weight_decay=.01,foreach=False)
def lr_factor(completed):
    # LambdaLR sets LR for the next update. Updates 1..15 linearly warm to peak.
    if completed<15: return (completed+1)/15
    phase=min(1.,(completed-15)/485)
    return .1+.9*.5*(1+math.cos(math.pi*phase))
def make_scheduler(opt): return torch.optim.lr_scheduler.LambdaLR(opt,lr_factor)

def update(model,opt,scheduler,group):
    denoms=denominator(group); opt.zero_grad(set_to_none=True)
    sums={k:0. for k in PARTS}
    for ex in group:
        loss,parts=sample_loss(model,ex,denoms)
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
        loss.backward()
        for k in PARTS:sums[k]+=parts[k]
    grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    lr=opt.param_groups[0]['lr'];opt.step();scheduler.step()
    return {'loss':sum(WEIGHTS[k]*sums[k]/denoms[k] for k in PARTS if denoms[k]),
            'component_ce':{k:sums[k]/denoms[k] if denoms[k] else 0. for k in PARTS},
            'supervised_tokens':denoms,'grad_norm':float(grad),'lr':lr}

def save_checkpoint(model,tok,opt,scheduler,path,completed,order,config):
    path=Path(path);assert not path.exists(),str(path)
    path.mkdir(parents=True)
    model.save_pretrained(path,safe_serialization=True,max_shard_size='10GB')
    tok.save_pretrained(path)
    state={'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'completed_updates':completed,
           'next_data_group':completed,'order':order,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
           'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'config':config}
    tmp=path/'resume.pt.tmp';torch.save(state,tmp);os.replace(tmp,path/'resume.pt')
    manifest={'utc':utc(),'completed_updates':completed,'next_data_group':completed,'parent':str(B0),'config':config,
              'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in path.iterdir() if p.is_file()}}
    write_json(path/'manifest.json',manifest)
    return manifest

def restore_training(path):
    model,tok=load_model(path,training=True)
    state=torch.load(Path(path)/'resume.pt',map_location='cpu',weights_only=False)
    opt=make_optimizer(model);opt.load_state_dict(state['optimizer'])
    scheduler=make_scheduler(opt);scheduler.load_state_dict(state['scheduler'])
    # Scheduler construction resets lr; restore its recorded last value explicitly.
    for group,lr in zip(opt.param_groups,scheduler.get_last_lr()):group['lr']=lr
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    return model,tok,opt,scheduler,state
