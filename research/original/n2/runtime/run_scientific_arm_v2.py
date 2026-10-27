"""Bounded scientific-arm launcher; locks the existing GPU schedule and records exit."""
import os,sys,pathlib,json,hashlib,datetime,subprocess
ROOT=pathlib.Path('/workspace/Nyqo');N=ROOT/'n2';RT=N/'runtime'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sh(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def put(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def main():
 assert len(sys.argv)==2 and sys.argv[1] in ['R','C']
 arm=sys.argv[1];d=RT/('scientific_'+arm+'1600_v2');f=json.loads((N/'data/train1600/FREEZE_RECEIPT_v2.json').read_text());p=json.loads((d/'prepared.json').read_text())
 assert p['status']=='PREPARED_NOT_EXECUTED' and p['branch']==arm
 assert f['version']==2 and f['optimizer_RNG_fresh_not_resume'] and f['budget']['approved_total_active_response_EOS_tokens_all_attempts']==643862
 assert f['budget']['already_spent_quarantined_R_v1_tokens']==214858
 if arm=='C':
  prior=RT/'scientific_R1600_v2';pr=json.loads((prior/'completed.json').read_text());ex=json.loads((prior/'process_exit.json').read_text());assert ex['returncode']==0 and pr['data_sha256']==f['data_sha256'] and pr['actual_active_tokens']==214502 and pr['frozen_student_unchanged'] and pr['completed_updates']==200
 assert p['data_sha256']==sh(p['data_path'])==f['data_sha256']
 assert sh(p['order_path'])==f['order_file_sha256']
 assert p['summary']==f['runtime_summary'] and p['scheduler_horizon']==200 and p['warmup']==6
 assert sh(RT/'n2_runtime.py')==f['runtime_sha256'] and sh(RT/'train.py')==f['driver_sha256']
 assert not (d/'launch.json').exists() and not (d/'updates.jsonl').exists() and not (d/'checkpoints').exists()
 cmd=['/usr/bin/flock','-n',str(N/'gpu.lock'),str(ROOT/'.venv/bin/python'),str(RT/'train.py'),'--rows',p['data_path'],'--data-sha256',p['data_sha256'],'--order',p['order_path'],'--run-dir',str(d),'--branch',arm,'--warmup','6','--stage-token-budget','300000','--execute']
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
 rec={'utc':utc(),'status':'SCIENTIFIC_TRAINING_STARTING_QUALITY_NOT_ASSESSED','branch':arm,'monitor_pid':os.getpid(),'argv':cmd,'data_sha256':p['data_sha256'],'order_object_sha256':p['order_sha256'],'freeze_receipt_sha256':sh(N/'data/train1600/FREEZE_RECEIPT_v2.json'),'driver_sha256':sh(RT/'train.py'),'runtime_sha256':sh(RT/'n2_runtime.py'),'prepared_sha256':sh(d/'prepared.json'),'planned_rows':1600,'planned_active_tokens':214502,'planned_updates':200,'fresh_optimizer':True,'quarantined_R_parent_used':False,'corrected_version':2,'approved_all_attempts_token_cap':643862,'prior_quarantined_R_v1_tokens':214858,'technical_parent_used':False,'gpu_lock_path':str(N/'gpu.lock')}
 put(d/'launch.json',rec)
 with (d/'training.log').open('xb') as log:
  child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
  rec.update(status='SCIENTIFIC_TRAINING_PROCESS_STARTED_QUALITY_NOT_ASSESSED',child_pid=child.pid);put(d/'launch.json',rec)
  result=child.wait()
 exitrec={'utc':utc(),'branch':arm,'returncode':result,'status':'COMPLETE_QUALITY_NOT_ASSESSED' if result==0 and (d/'completed.json').exists() else 'FAILED_REQUIRES_DIAGNOSIS','child_pid':child.pid,'completed_summary':str(d/'completed.json') if (d/'completed.json').exists() else None}
 put(d/'process_exit.json',exitrec)
if __name__=='__main__':main()
