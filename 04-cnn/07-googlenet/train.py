import torch
from torch import nn
import logging
import json

class Writer:
    def __init__(self, fname: str):
        self.fname = fname
        self.metrics = {}


    def add(self, step, metric_name, value):
        if not metric_name in self.metrics:
            self.metrics[metric_name] = {}
        self.metrics[metric_name][step] = value

    def save(self):
        with open(self.fname+".json", "w") as f:
            json.dump(self.metrics, f)

def train(device, model, data, optimizer, scheduler, max_epochs, log_every=10):

    model_name = "googlenet"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f"training-{model_name}.log")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Start training of {model_name}")
    logger.info(f"Device {device}")
    logger.info(f"Train dataset: {data.train} - {len(data.train)} instances")
    logger.info(f"Eval dataset: {data.eval} - {len(data.eval)} instances")
    logger.info(f"Model: {model}")
    logger.info(f"Optimizer: {optimizer}")


    criterion = nn.CrossEntropyLoss()
    train_dataloader=data.train_dataloader
    eval_dataloader=data.eval_dataloader

    overall_metrics = {
        "train_loss": [],
        "eval_loss": [],
        "train_accuracy": [],
        "eval_accuracy": [],
        "train_step": [],
        "eval_step": []
    }

    best_acc = 0

    writer = Writer(fname=model_name)
    setattr(writer, "num_train_batches", len(train_dataloader))
    setattr(writer, "max_epochs", max_epochs)
    
    writer.metrics["num_train_batches"] = len(train_dataloader)
    writer.metrics["max_epochs"] = max_epochs


    def fit_epoch(epoch_num):
        train_loss = 0.
        eval_loss = 0.
        train_acc = 0.
        eval_acc = 0.
        train_instances = 0
        eval_instances = 0

        model.train()


        for step_num, batch in enumerate(train_dataloader):
            batch = [a.to(device) for a in batch]
            main_output, aux_output1, aux_output2 = model(*batch[:-1])

            optimizer.zero_grad()
            main_loss = criterion(main_output, batch[-1])
            aux_loss2 = criterion(aux_output1, batch[-1])
            aux_loss3 = criterion(aux_output2, batch[-1])
            loss = main_loss + 0.3 * aux_loss2 + 0.3 * aux_loss3

            loss.backward()
            optimizer.step()

            train_instances += batch[-1].size(0)
            train_loss += main_loss.item() * batch[-1].size(0)
            acc = (batch[-1]==main_output.argmax(dim=-1)).float().mean() * batch[-1].size(0)
            train_acc += acc.item()
            avg_loss = train_loss / train_instances
            avg_acc = train_acc / train_instances


            overall_metrics["train_loss"].append(avg_loss)
            overall_metrics["train_accuracy"].append(avg_acc)
            global_step = (epoch_num*len(train_dataloader)) + step_num
            overall_metrics["train_step"].append(global_step)

            writer.add(step=global_step, metric_name="Loss/train", value=avg_loss)
            writer.add(step=global_step, metric_name="Accuracy/train", value=avg_acc)

            if step_num % log_every == 0:
                logger.info(f"[Epoch {epoch_num+1}/{max_epochs}] [Step {global_step}] train_loss: {avg_loss:.5f}, train_acc: {avg_acc:.5f}")


        model.eval()

        for step_num, batch in enumerate(eval_dataloader):
            batch = [a.to(device) for a in batch]
            with torch.no_grad():
                main_output, _, _= model(*batch[:-1])
                loss = criterion(main_output, batch[-1])

                eval_instances += batch[-1].size(0)
                eval_loss += loss.item() * batch[-1].size(0)
                acc = (batch[-1]==main_output.argmax(dim=-1)).float().mean() * batch[-1].size(0)
                eval_acc += acc.item()
                avg_loss = eval_loss / eval_instances
                avg_acc = eval_acc / eval_instances

                overall_metrics["eval_loss"].append(avg_loss)
                overall_metrics["eval_accuracy"].append(avg_acc)
                global_step = (epoch_num*len(eval_dataloader)) + step_num
                overall_metrics["eval_step"].append(global_step)

        train_loss = train_loss / train_instances
        train_acc = train_acc / train_instances
        eval_loss = eval_loss / eval_instances
        eval_acc = eval_acc / eval_instances

        writer.add(step=epoch_num, metric_name="Loss/eval", value=eval_loss)
        writer.add(step=epoch_num, metric_name="Accuracy/eval", value=eval_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"[Epoch {epoch_num+1}/{max_epochs}] train_loss: {train_loss:.5f}, train_acc: {train_acc:.5f}, eval_loss: {eval_loss:.5f}, eval_acc: {eval_acc:.5f}, lr: {current_lr:.5f}")
        scheduler.step()

        return eval_acc

    for epoch_num in range(max_epochs):
        eval_acc = fit_epoch(epoch_num)
        torch.save(model.state_dict(), f"{model_name}.pth")
        logger.info(f"[Epoch {epoch_num+1}/{max_epochs}] Model saved at {model_name}.pth")

        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(model.state_dict(), f"best-{model_name}-net.pth")
            
    writer.save()
    return overall_metrics
