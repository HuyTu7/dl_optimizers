import pickle
import numpy as np
import matplotlib as plt

def get_stats(domain=""):
    optimizers = ["adam", "lamb", "lars", "sgd"]
    if domain == "image_classification":
        batch_size = [512, 1024, 4096, 8192]
        for bs in batch_size:
            for op in optimizers:
                fname = "%s/results/%s_%s_metrics.p" % (domain, op, bs)
                metrics = pickle.load(open(fname, "rb"))
                test_losses = np.array(metrics['test_loss'])
                print(op, bs, np.min(test_losses), np.where(test_losses == np.min(test_losses)))

            print("*"*10)
    else:
        batch_size = [64, 128, 256, 512]
        for bs in batch_size:
            for op in optimizers:
                fname = "%s/results/train_%s_%s/ckpt/metrics.p" % (domain, op, bs)
                metrics = pickle.load(open(fname, "rb"))
                test_losses = np.array(metrics['valid_losses'])
                print(op, bs, np.min(test_losses), np.where(test_losses == np.min(test_losses)))

            print("*"*10)



if __name__ == '__main__':
    get_stats("qa_system")