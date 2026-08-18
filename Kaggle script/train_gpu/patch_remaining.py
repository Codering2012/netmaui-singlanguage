import re

with open('dataset.py', 'r') as f:
    text = f.read()

# Fix max_workers=2 in ThreadPoolExecutor
text = text.replace('with ThreadPoolExecutor(max_workers=2) as executor:', 'with ThreadPoolExecutor(max_workers=1) as executor:')

with open('dataset.py', 'w') as f:
    f.write(text)
print("dataset.py patched")


with open('train_all_in_one_tpu.py', 'r') as f:
    text = f.read()
    
# Fix Validation ParallelLoader retention
old_ret = """    return {
        "loss": avg_loss,"""
new_ret = """    if 'para_loader' in locals():
        del para_loader
    import gc
    gc.collect()

    return {
        "loss": avg_loss,"""
text = text.replace(old_ret, new_ret)

with open('train_all_in_one_tpu.py', 'w') as f:
    f.write(text)
    
print("train patched")
