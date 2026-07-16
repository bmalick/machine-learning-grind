
def compare_plots(self, metrics_a, metrics_b, other_trainer):
    for pername in ["perstep", "perepoch"]:
        for loss_name in metrics_a[pername]:
            fig, ax = plt.subplots(figsize=self.config.figsize)
            for (split1,value1), (split2,value2) in zip(metrics_a[pername][loss_name].items(), metrics_b[pername][loss_name].items()):
                if self.config.figlog:
                    if len(value1):
                        ax.semilogy(value1, label=f"{self.config.run_name} {loss_name}/{split1}", linestyle="-" if split1=="train" else "--")
                    if len(value2):
                        ax.semilogy(value2, label=f"{other_trainer.config.run_name} {loss_name}/{split2}", linestyle="-" if split2=="train" else "--")
                else:
                    if len(value1):
                        ax.plot(value1, label=f"{self.config.run_name} {loss_name}/{split1}", linestyle="-" if split1=="train" else "--")
                    if len(value2):
                        ax.plot(value2, label=f"{other_trainer.config.run_name} {loss_name}/{split2}", linestyle="-" if split2=="train" else "--")
                if self.config.figgrid: ax.grid()
            ax.legend()
            ax.set_xlabel(pername.replace("per","")+"s")
            ax.set_title(loss_name)
            compare_dir = f"{self.config.run_id}-vs-{other_trainer.config.run_id}"
            os.makedirs(compare_dir, exist_ok=True)
            fig.savefig(os.path.join(compare_dir, f"{pername}-{loss_name}.jpg"))
            plt.close()


# def load_checkpoint(logdir: str, device="cpu"):
#     ckpt_fname = os.path.join(logdir, "ckpt.pt")
#     checkpoint = torch.load(ckpt_fname, map_location=device)
#     return checkpoint["train_config"]["run_id"], checkpoint["history"]

def load_checkpoint(logdir: str):
    run_id = os.path.basename(logdir).split("--")[0]
    history = json.load(open(os.path.join(logdir, "history.json"), "r"))
    return run_id, history

def compare_plots(logdir1, logdir2, figsize=(8., 4.5)):
    run_id1, history1 = load_checkpoint(logdir1)
    run_id2, history2 = load_checkpoint(logdir2)
    for pername in ["perstep", "perepoch"]:
        for loss_name in history1[pername]:
            fig, ax = plt.subplots(figsize=figsize)
            for (split1,value1), (split2,value2) in zip(history1[pername][loss_name].items(), history2[pername][loss_name].items()):
                if value1:
                    ax.plot(value1, label=f"{run_id1} {loss_name}/{split1}", linestyle="-" if split1=="train" else "--")
                if value2:
                    ax.plot(value2, label=f"{run_id2} {loss_name}/{split2}", linestyle="-" if split2=="train" else "--")
            ax.grid()
            ax.legend()
            ax.set_xlabel(pername.replace("per","")+"s")
            ax.set_title(loss_name)
            compare_dir = f"{run_id1}-vs-{run_id2}"
            os.makedirs(compare_dir, exist_ok=True)
            fig.savefig(os.path.join(compare_dir, f"{pername}-{loss_name}.jpg"))
            plt.close()
