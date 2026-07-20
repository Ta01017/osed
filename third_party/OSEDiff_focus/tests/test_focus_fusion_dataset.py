import json
import numpy as np
from PIL import Image
import pytest
from dataloaders.focus_fusion_dataset import FocusFusionDataset, FIXED_FUSION_PROMPT

def _files(tmp_path,n=3):
    records=[]
    for i in range(n):
        names=[]
        for j,mode in enumerate(("RGB","RGB","RGB","L","L")):
            name=f"{i}_{j}.png"; value=32+i*20+j
            a=np.full((12,16,3) if mode=="RGB" else (12,16),value,np.uint8); Image.fromarray(a,mode).save(tmp_path/name); names.append(name)
        records.append({"image":names[0],"edit_image":names[1:],"prompt":f"p{i}"})
    meta=tmp_path/"metadata.json"; meta.write_text(json.dumps(records),encoding="utf-8"); return meta

def test_dataset_ranges_shapes_and_smoke(tmp_path):
    meta=_files(tmp_path,20); ds=FocusFusionDataset(meta,tmp_path,resolution=8,smoke=True,prompt_mode="fixed")
    assert len(ds)==16; x=ds[0]
    assert x["gt"].shape==x["a"].shape==x["b_warp"].shape==(3,8,8)
    assert x["focus_a"].shape==x["focus_b_warp"].shape==(1,8,8)
    assert -1<=x["a"].min()<=x["a"].max()<=1 and 0<=x["focus_a"].min()<=x["focus_a"].max()<=1
    assert x["prompt"]==FIXED_FUSION_PROMPT

def test_start_limit_metadata_prompt_and_absolute_paths(tmp_path):
    meta=_files(tmp_path); ds=FocusFusionDataset(meta,tmp_path,8,max_samples=2,start_index=1,prompt_mode="metadata")
    assert len(ds)==2 and ds[0]["metadata_index"]==1 and ds[0]["prompt"]=="p1"

def test_missing_path_reports_index_and_full_path(tmp_path):
    meta=tmp_path/"m.json"; meta.write_text(json.dumps([{"image":"missing.png","edit_image":["a","b","fa","fb"]}]))
    with pytest.raises(FileNotFoundError,match=r"metadata index 0: missing path: .*missing.png"): FocusFusionDataset(meta,tmp_path,8)[0]

