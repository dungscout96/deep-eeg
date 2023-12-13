import torch

class NNUtils():
    def randomize_weights(self, rand_params):
        def zero_weights(m):
            # TODO this might need expansion depending on model
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                m.weight.data.fill_(0.)
                if hasattr(m, "bias"):
                    m.bias.data.fill_(0.)
        def init_weights(m):
            # TODO this might need expansion depending on model
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                m.reset_parameters()

        if rand_params["mode"] == "zero":
            self.model.apply(weight_reset)

        elif rand_params["mode"] == "rand_init":
            if "seed" in rand_params:
                torch.manual_seed(rand_params["seed"])
            self.model.apply(init_weights)

        elif rand_params["mode"] == "perturb":
            for idx, (name, param) in enumerate(self.model.named_parameters()):
                noise = torch.zeros_like(param)
                val_range = (torch.max(param)-torch.min(param)).item()
                if rand_params["distribution"] == "gaussian":
                    torch.nn.init.normal_(noise, 0, val_range/100)
                elif rand_params["distribution"] == "uniform":
                    torch.nn.init.uniform_(noise, val_range/200, val_range/100)
                else:
                    raise(f"Distribution {rand_params['distribution']} not supported")
                with torch.no_grad():
                    param += noise

        elif rand_params["mode"] == "shuffle":
            for idx, (name, param) in enumerate(self.model.named_parameters()):
                ori_shape = param.shape
                with torch.no_grad():
                    flattened = param.flatten()
                    flattened = flattened[torch.randperm(len(flattened))]
                    param.copy_(torch.reshape(flattened, ori_shape))

        else:
            raise ValueError(f"Unsupported random mode {rand_params['mode']}")
