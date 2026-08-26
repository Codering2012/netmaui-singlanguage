import glob
import torch

def main():
    shards = sorted(glob.glob('E:/datasets/asl_dataset/asl_preprocessed_phase1/train/shard_*.pt'))[:15]
    print("Inspecting first 15 shards...")
    for s in shards:
        try:
            d = torch.load(s, map_location='cpu', weights_only=False)
            shapes = [item['features'].shape[0] for item in d if isinstance(item, dict) and 'features' in item]
            unique_lens = set(shapes)
            tasks = set(item.get('task', 'none') for item in d if isinstance(item, dict))
            sources = set(item.get('source', 'none') for item in d if isinstance(item, dict))
            labels_sample = [item.get('label') for item in d[:5] if isinstance(item, dict)]
            print(f"Shard {s.split('/')[-1].split(chr(92))[-1]}: {len(d)} items | Frame lengths: {unique_lens} | Tasks: {tasks} | Sources: {sources} | Sample labels: {labels_sample}")
        except Exception as e:
            print(f"Error {s}: {e}")

if __name__ == "__main__":
    main()
