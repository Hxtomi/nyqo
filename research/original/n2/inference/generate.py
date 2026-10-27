"""N2 single-GPU free generation. Receipts record execution, never quality."""
import argparse, datetime as dt, fcntl, hashlib, importlib.metadata as md, json, os, sys, time
from pathlib import Path
ROOT=Path("/workspace/Nyqo")
N2=ROOT/"n2"
os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
 HF_HUB_DISABLE_TELEMETRY="1",TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",
 HF_HOME=str(N2/".cache/hf"),TMPDIR=str(N2/"tmp"))
sys.path.insert(0,str(ROOT/"n1"))
import runtime as base
import expanded_runtime as expanded
import torch
PROFILE={"dtype":"BF16","attention":"sdpa","cache":True,"do_sample":False,"num_beams":1,
 "max_new_tokens":768,"max_total_tokens":2048,"eos_token_id":7,"pad_token_id":0,"batch_size":1,
 "template":"native B0 add_generation_prompt=True","parser":"n1.runtime.parse_answer","seed":base.SEED}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def write_json(path,data):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+".tmp")
 tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
 tmp.replace(p)
def rows_from(path):
 text=path.read_text(encoding="utf-8")
 rows=json.loads(text) if text.lstrip().startswith("[") else [json.loads(x) for x in text.splitlines() if x.strip()]
 assert len({r["id"] for r in rows})==len(rows)
 for r in rows:
  assert r.get("language","en")=="en" and r["messages"][-1]["role"]=="user"
  assert all(set(m)<= {"role","content"} and m["role"] in ("system","user","assistant") for m in r["messages"])
 return rows
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--input",required=True);p.add_argument("--output",required=True)
 p.add_argument("--model",choices=["B0","K"],required=True)
 p.add_argument("--model-path");p.add_argument("--run-index",type=int,default=1)
 a=p.parse_args()
 lock=(N2/"gpu.lock").open("a")
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 bank=Path(a.input).resolve();out=Path(a.output).resolve()
 assert bank.is_relative_to(ROOT) and out.is_relative_to(ROOT)
 assert "private_final" not in str(bank), "FINAL generation must be separately scheduled by evaluation owner"
 rows=rows_from(bank)
 modelpath=Path(a.model_path) if a.model_path else (base.B0 if a.model=="B0" else ROOT/"models/E497K-update0500")
 profile=dict(PROFILE,software={n:md.version(n) for n in ["torch","transformers","tokenizers"]})
 profile_sha=hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest()
 cfg={"profile":profile,"profile_sha256":profile_sha,"input_sha256":sha(bank),"model_sha256":sha(modelpath/"model.safetensors"),
  "model_path":str(modelpath),"tokenizer_sha256":sha(base.B0/"tokenizer.json"),"template_sha256":sha(base.B0/"chat_template.jinja"),
  "generator_sha256":sha(__file__),"runtime_sha256":sha(ROOT/"n1/runtime.py"),"expanded_runtime_sha256":sha(ROOT/"n1/expanded_runtime.py"),
  "run_index":a.run_index,"plan_commit":(N2/"CANON_COMMIT").read_text().strip()}
 manifest=out.with_suffix(".manifest.json")
 if manifest.exists():assert json.loads(manifest.read_text())==cfg,"generation manifest changed"
 else:write_json(manifest,cfg)
 done={}
 if out.exists():
  for line in out.read_text().splitlines():
   r=json.loads(line)
   assert r["input_sha256"]==cfg["input_sha256"] and r["model_sha256"]==cfg["model_sha256"] and r["profile_sha256"]==profile_sha
   assert r["id"] not in done
   done[r["id"]]=r
 base.seed_all()
 model,tok=(base.load_model(modelpath,training=False) if a.model=="B0" and modelpath==base.B0 else expanded.load_model(modelpath,training=False))
 model.eval()
 print(json.dumps({"event":"generation_start","pid":os.getpid(),"model":a.model,"rows":len(rows),"completed":len(done)}),flush=True)
 out.parent.mkdir(parents=True,exist_ok=True)
 for row in rows:
  if row["id"] in done:continue
  prompt=tok.apply_chat_template(row["messages"],tokenize=False,add_generation_prompt=True)
  inp=tok(prompt,add_special_tokens=False,return_tensors="pt").to("cuda")
  n=inp.input_ids.shape[1]
  assert n+PROFILE["max_new_tokens"]<=PROFILE["max_total_tokens"],(row["id"],"input budget",n)
  torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
  with torch.inference_mode():
   output=model.generate(**inp,do_sample=False,num_beams=1,max_new_tokens=768,use_cache=True,eos_token_id=7,pad_token_id=0)
  torch.cuda.synchronize()
  elapsed=time.monotonic()-start
  new=output[0,n:].tolist();raw=tok.decode(new,skip_special_tokens=False);parsed=base.parse_answer(raw)
  rec={"id":row["id"],"domain":row["domain"],"utc":dt.datetime.now(dt.timezone.utc).isoformat(),
   "run_index":a.run_index,"input_sha256":cfg["input_sha256"],"model_sha256":cfg["model_sha256"],"profile_sha256":profile_sha,
   "input_tokens":n,"input_ids_sha256":hashlib.sha256(json.dumps(inp.input_ids[0].tolist()).encode()).hexdigest(),
   "output_tokens":len(new),"token_ids":new,"raw_text":raw,**parsed,"eos_reached":bool(new and new[-1]==7),
   "cutoff":not bool(new and new[-1]==7),"seconds":elapsed,"peak_bytes":torch.cuda.max_memory_allocated()}
  with out.open("a",encoding="utf-8") as f:
   f.write(json.dumps(rec,ensure_ascii=False)+"\n");f.flush();os.fsync(f.fileno())
  done[row["id"]]=rec
  print(json.dumps({"event":"generation_row","id":row["id"],"completed":len(done),"tokens":len(new),"seconds":round(elapsed,3),"cutoff":rec["cutoff"]}),flush=True)
  del output,inp
 summary={"status":"GENERATION_COMPLETE_UNGRADED","utc":dt.datetime.now(dt.timezone.utc).isoformat(),"rows":len(done),
  "seconds":sum(r["seconds"] for r in done.values()),"output_tokens":sum(r["output_tokens"] for r in done.values()),
  "cutoffs":sum(r["cutoff"] for r in done.values()),"output_sha256":sha(out)}
 write_json(out.with_suffix(".complete.json"),summary)
 print(json.dumps(summary),flush=True)
if __name__=="__main__":main()
