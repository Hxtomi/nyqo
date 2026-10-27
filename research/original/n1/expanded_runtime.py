"""Native E497 extension. Original N1 runtime/objective/parser remain immutable."""
import copy
from runtime import *

NEW=(4,5,10,11,16,17,22,23)
TOTAL=497119488
TRAINABLE=142635520

def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def frozen_hashes(model):
    return {n:tensor_hash(p) for n,p in model.named_parameters() if not p.requires_grad}

def layout():
    result=[]
    for old in range(16):
        result.append({'new_index':len(result),'source_index':old,'trainable':False})
        if old in (3,7,11,15):
            for source in (14,15):result.append({'new_index':len(result),'source_index':source,'trainable':True})
    return result

def configure_training(model,training):
    for p in model.parameters():p.requires_grad_(False)
    for index in NEW:
        for p in model.model.layers[index].parameters():
            p.requires_grad_(training)
            if training:p.data=p.data.float()
    assert sum(p.numel() for p in model.parameters())==TOTAL
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==(TRAINABLE if training else 0)
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert model.get_input_embeddings().weight.dtype==torch.bfloat16
    model.config.use_cache=not training
    model.train(training)
    return model

def build_identity(training=False,device='cuda'):
    model=AutoModelForCausalLM.from_pretrained(str(B0),local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa')
    assert sum(p.numel() for p in model.parameters())==354483968
    original=list(model.model.layers);types=list(model.config.layer_types)
    pair=[copy.deepcopy(original[i]) for i in (14,15)]
    assert sum(p.numel() for layer in pair for p in layer.parameters())==35658880
    layers=[];newtypes=[];zeros=[]
    for entry in layout():
        i=entry['new_index'];source=entry['source_index']
        layer=copy.deepcopy(pair[source-14]) if entry['trainable'] else original[source]
        if entry['trainable']:
            projections=[('self_attn.out_proj' if layer.is_attention_layer else 'conv.out_proj',layer.self_attn.out_proj if layer.is_attention_layer else layer.conv.out_proj),('feed_forward.w2',layer.feed_forward.w2)]
            with torch.no_grad():
                for name,module in projections:
                    module.weight.zero_();zeros.append(f'model.layers.{i}.{name}.weight')
                    if module.bias is not None:module.bias.zero_();zeros.append(f'model.layers.{i}.{name}.bias')
        operator=layer.self_attn if layer.is_attention_layer else layer.conv
        operator.layer_idx=i
        layers.append(layer);newtypes.append(types[source])
    cfg=copy.deepcopy(model.config);cfg.num_hidden_layers=24;cfg.layer_types=newtypes
    cfg.full_attn_idxs=[i for i,t in enumerate(newtypes) if t=='full_attention']
    cfg.nyqo_expansion={'parent_revision':B0.name,'layout':layout(),'zeroed_tensors':zeros,'new_positions':list(NEW),'total_parameters':TOTAL,'trainable_parameters':TRAINABLE,'initialization':'independent clean-B0 14/15 copies; actual residual outputs zero'}
    model.config=cfg;model.model.config=cfg;model.model.rotary_emb.config=cfg
    model.model.layers=torch.nn.ModuleList(layers)
    for layer in layers:
        operator=layer.self_attn if layer.is_attention_layer else layer.conv
        if hasattr(operator,'config'):operator.config=cfg
    pointers=[p.data_ptr() for layer in layers for p in layer.parameters()]
    assert len(pointers)==len(set(pointers)),'Unexpected shared layer storage'
    configure_training(model,training);model.to(device)
    tok=AutoTokenizer.from_pretrained(str(B0),local_files_only=True,trust_remote_code=False)
    return model,tok

def load_model(path=B0,training=False,attention='sdpa'):
    assert attention=='sdpa'
    if Path(path)==B0:return build_identity(training=training)
    model=AutoModelForCausalLM.from_pretrained(str(path),local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation=attention)
    assert model.config.num_hidden_layers==24
    assert model.config.nyqo_expansion['new_positions']==list(NEW)
    configure_training(model,training)
    if training:
        from safetensors.torch import load_file
        state=load_file(str(Path(path)/'model.safetensors'))
        incompatible=model.load_state_dict(state,strict=False)
        assert not incompatible.unexpected_keys
        assert set(incompatible.missing_keys)<= {'lm_head.weight'}
        del state
    for i,layer in enumerate(model.model.layers):
        assert (layer.self_attn if layer.is_attention_layer else layer.conv).layer_idx==i
    model.to('cuda');return model,AutoTokenizer.from_pretrained(str(B0),local_files_only=True,trust_remote_code=False)

def restore_training(path):
    model,tok=load_model(path,training=True)
    state=torch.load(Path(path)/'resume.pt',map_location='cpu',weights_only=False)
    assert state['config']['branch']=='E497'
    opt=make_optimizer(model);opt.load_state_dict(state['optimizer'])
    scheduler=make_scheduler(opt);scheduler.load_state_dict(state['scheduler'])
    for group,lr in zip(opt.param_groups,scheduler.get_last_lr()):group['lr']=lr
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    return model,tok,opt,scheduler,state
