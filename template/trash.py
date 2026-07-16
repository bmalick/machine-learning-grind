
    def train_step(self, epoch_num):
        self.model.train()
        accum_steps = self.config.grad_accum_steps

        epoch_loss = 0.
        num_instances = 0
        epoch_metrics = {n: 0. for n in self.metric_names}
        accum_metrics = {n: 0. for n in self.metric_names}
        n_dataloader = len(self.datamodule.train_dataloader)

        self.optimizer.zero_grad()

        for step_num, batch in enumerate(self.datamodule.train_dataloader):
            batch = self.to_device(batch)
            out, loss = self.model(*batch[:-1], batch[-1])

            step_metrics = self.compute_metrics(out, batch[-1])
            for k,v in step_metrics.items():
                accum_metrics[k] += v / accum_steps

            loss /= accum_steps
            loss.backward()

            is_last_batch = (step_num+1 == n_dataloader)

            if (step_num + 1) % accum_steps == 0 or is_last_batch:
                if self.scheduler is not None:
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.scheduler(epoch_num)
                # gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip)
                # self.losses["grad_norm"]["train"].append(gn.item())
                self.optimizer.step()
                self.optimizer.zero_grad()

                bs = batch[-1].size(0)
                num_instances += bs
                true_loss = loss.item() * accum_steps
                epoch_loss += true_loss * bs

                it =  epoch_num * n_dataloader + step_num
                for k,v in accum_metrics.items():
                    self.losses["perstep"][k]["train"].append(v)
                    self.writer.add_scalar(f"perstep_{k}/train", v, it)
                    epoch_metrics[k] += v * bs

                self.losses["perstep"]["loss"]["train"].append(true_loss)
                self.writer.add_scalar("perstep_loss/train", true_loss, it)

                accum_metrics = {n: 0. for n in self.metric_names}

        for k,v in epoch_metrics.items():
            self.losses["perepoch"][k]["train"].append(v/num_instances)
            self.writer.add_scalar(f"perepoch_{k}/train", v/num_instances, epoch_num)

        epoch_loss /= num_instances
        self.losses["perepoch"]["loss"]["train"].append(epoch_loss)
        self.writer.add_scalar("perepoch_loss/train", epoch_loss, epoch_num)
