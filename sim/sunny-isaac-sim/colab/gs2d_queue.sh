#!/bin/bash
# Waits for the gsplat build, smoke-tests 2DGS (tyro flags, 300 steps, depth export at 5 cm voxels; ends SMOKE_OK or SMOKE_FAIL),
# then after TRI_DONE and PM_DONE (GPU free of 3DGUT) runs gs2d_run.sh, read fresh so it can be patched until then.
# Usage: nohup bash gs2d_queue.sh > /content/gs2d_queue.log 2>&1 &   (gs2d_run output in the same log, ends GS2D_DONE)
while pgrep -f '[u]v pip install --no-build' > /dev/null; do sleep 30; done
cd /content/gsplat/examples
python3 -c 'from gsplat.rendering import rasterization_2dgs; import gsplat; print("GS_OK", gsplat.__version__)' || { echo SMOKE_FAIL build; exit 1; }
python3 simple_trainer_2dgs.py --help 2>&1 | grep -i -E "normalize|disable.viewer|normal.loss|dist.loss" | head
S=/content/gs2d_smoke; G=/content/gs_da; D=/content/eth3d/delivery_area
mkdir -p $G/sparse $S
[ -e $G/sparse/0 ] || cp -r /content/grut_da/sparse/0 $G/sparse/0
[ -e $G/images ] || ln -s $D/images $G/images
[ -e $G/images_3 ] || ln -s /content/grut_da/images_3 $G/images_3
# gsplat resizes the full JPEGs to PNG but does not create the subfolder in the image names
mkdir -p $G/images_3_png/dslr_images_undistorted
python3 simple_trainer_2dgs.py --data_dir $G --data_factor 3 --result_dir $S --no-normalize_world_space --disable_viewer \
  --normal_loss --dist_loss --max_steps 300 --eval_steps 300 --save_steps 300 2>&1 | grep -v "it/s\]" | tail -15
ls $S/ckpts $S/stats; cat $S/stats/val_step299.json
python3 /content/code/gs2d_depth.py render $S/ckpts/ckpt_299.pt $G $S 2>&1 | grep -v "^view" | tail -8
/content/o3d19/bin/python /content/code/gs2d_depth.py fuse $S 0.05 2>&1 | tail -3
[ -s $S/mesh_points.ply ] && echo SMOKE_OK || { echo SMOKE_FAIL; exit 1; }
until grep -q TRI_DONE /content/walk_tri.log 2>/dev/null && grep -q PM_DONE /content/pm_da.log 2>/dev/null; do sleep 60; done
bash /content/gs2d_run.sh
