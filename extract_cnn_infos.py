from accelerator_util.get_flops import get_flops

from models import CNN, get_models

with open("out.txt","w") as f:
    for bs in [1, 32]:
        f.write(f"=== FIXED BATCH_SIZE= {bs} === \n")
        for model_code_name in get_models().keys():
            def F():
                m=CNN(model_code_name,fixed_batch_size=bs)
                return m

            info = get_flops(F)

            info_tuple=f"{model_code_name} {bs} {info[0]} {info[1]} {float(info[0]) / info[1]} \n"
            f.write(info_tuple)