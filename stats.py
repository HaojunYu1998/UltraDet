import numpy as np
import torch

rec16_list = []
prec80_list = []
prec90_list = []
ap50_list = []
fp80_list = []
fp90_list = []

for fold in range(1, 4):
    # for r in range(1,4):
    lines = open(f"miccai_outputs/UltraDet/UltraDet_Attn/fold{fold}_iter5k_r1/log.txt").readlines()
    lines = lines[-20:]
    for i, line in enumerate(lines):
        if "d2.evaluation.testing INFO: copypaste: R@16" in line:
            rec16 = float(lines[i+1].strip().split(" ")[-1])
            rec16_list.append(rec16 * 100)
        if "d2.evaluation.testing INFO: copypaste: P@R0.7,P@R0.8,P@R0.9" in line:
            _, prec08, prec09 = lines[i+1].strip().split(" ")[-1].split(",")
            prec80_list.append(float(prec08) * 100)
            prec90_list.append(float(prec09) * 100)
        if "d2.evaluation.testing INFO: copypaste: mAP,AP50,AP75" in line:
            ap50 = float(lines[i+1].strip().split(" ")[-1].split(",")[1])
            ap50_list.append(ap50 * 100)
        if "d2.evaluation.testing INFO: copypaste: FP@R0.7,FP@R0.8,FP@R0.9" in line:
            _, fp08, fp09 = lines[i+1].strip().split(" ")[-1].split(",")
            fp80_list.append(float(fp08))
            fp90_list.append(float(fp09))

# K = 4
# index = torch.tensor(fp90_list).topk(K, largest=False)[1].tolist()
# prec80_list = np.array(prec80_list)[index]
# prec90_list = np.array(prec90_list)[index]
# fp80_list = np.array(fp80_list)[index]
# fp90_list = np.array(fp90_list)[index]
# ap50_list = np.array(ap50_list)[index]
# rec16_list = np.array(rec16_list)[index]

print("prec80: ", np.mean(prec80_list), np.std(prec80_list))
print("prec90: ", np.mean(prec90_list), np.std(prec90_list))
print("fp80: ", np.mean(fp80_list), np.std(fp80_list))
print("fp90: ", np.mean(fp90_list), np.std(fp90_list))
print("ap50: ", np.mean(ap50_list), np.std(ap50_list))
print("rec16: ", np.mean(rec16_list), np.std(rec16_list))

print(f"{np.mean(prec80_list):.1f}\\tiny{{{np.std(prec80_list):.1f}}} & {np.mean(prec90_list):.1f}\\tiny{{{np.std(prec90_list):.1f}}} & {np.mean(fp80_list):.1f}\\tiny{{{np.std(fp80_list):.1f}}} & {np.mean(fp90_list):.1f}\\tiny{{{np.std(fp90_list):.1f}}} & {np.mean(ap50_list):.1f}\\tiny{{{np.std(ap50_list):.1f}}} & {np.mean(rec16_list):.1f}\\tiny{{{np.std(rec16_list):.1f}}}")