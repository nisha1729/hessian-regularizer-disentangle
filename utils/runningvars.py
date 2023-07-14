import os
import torch
import matplotlib.pyplot as plt

" code inspired by https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/8f27c9b97d2ca7c6e05333d5766d144bf7d8c31b/mit_semseg/utils.py#L33"


class RunningBaseVars:
    def __init__(self):
        self.cum_list = []
        self.cum_sum = 0.0
        self.mean = 0.0
        self.length = 0.0
        self.epoch = 0
        self.reset()

    def reset(self):
        # do not add c um_list here since it has to be in memory until end of training
        self.cum_sum = 0.0
        self.mean = 0.0
        self.length = 0.0

    def add(self, current, is_tensor=True):
        current = current.detach().cpu().numpy() if torch.is_tensor(current) is True else current

        self.cum_sum += current
        self.length += 1.

    def append(self, current):      # append mean at the end of each epoch
        self.cum_list.append(current)
        self.epoch += 1

    def average(self):
        self.mean = self.cum_sum / self.length


class RunningVars:
    def __init__(self, list_of_vars):
        super(RunningVars, self).__init__()
        """list_of_vars: lsit of strings of variables"""
        self.dict_of_vars = {var: RunningBaseVars() for var in list_of_vars}

    def append_all(self):
        [self.dict_of_vars[key].append(value.mean) for key, value in self.dict_of_vars.items()]

    def add_all(self, curr_values):
        [self.dict_of_vars[key].add(curr_val) for key, curr_val in zip(self.dict_of_vars, curr_values)]

    def average_all(self):
        [self.dict_of_vars[key].average() for key in self.dict_of_vars.keys()]

    def reset_all(self):
        [self.dict_of_vars[key].reset() for key in self.dict_of_vars.keys()]

    def print_all(self, mode: str):
        """Must be called after average_all() and append_all()"""
        if mode == 'Test':
            print(f'[===={mode}=====]')
            [print(f'{key} : {value.mean}') for key, value in self.dict_of_vars.items()]
            return

        print(f'Epoch: {list(self.dict_of_vars.values())[0].epoch} | [===={mode}=====]')
        [print(f'{key} : {value.mean : .6f}') for key, value in self.dict_of_vars.items()]


class GeneratePlots(RunningVars):
    """Created this class since I need train and val losses in the same plot for comparison"""
    def __init__(self, train_var: RunningVars, val_var: RunningVars, test_var: RunningVars):
        super(GeneratePlots, self).__init__([train_var, val_var, test_var])
        self.train_var = train_var
        self.val_var = val_var
        self.test_var = test_var

    def plot_all(self, filename):
        reg = True if 'Regulariser' in self.train_var.dict_of_vars.keys() else False
        compare = True if 'Hessian RSS' in self.train_var.dict_of_vars.keys() else False
        if reg or compare:
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
        else:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        fig.tight_layout()

        ax1.plot([x for x in range(self.train_var.dict_of_vars['Combined loss'].epoch)], self.train_var.dict_of_vars['Combined loss'].cum_list, label='Total train loss')
        ax1.plot([x for x in range(self.val_var.dict_of_vars['Combined loss'].epoch)], self.val_var.dict_of_vars['Combined loss'].cum_list, label='Total val loss')
        ax1.set_title('Combined (weighted) Loss')
        ax1.legend()

        ax2.plot([x for x in range(self.train_var.dict_of_vars['AE loss'].epoch)], self.train_var.dict_of_vars['AE loss'].cum_list, label='AE train loss')
        ax2.plot([x for x in range(self.val_var.dict_of_vars['AE loss'].epoch)], self.val_var.dict_of_vars['AE loss'].cum_list, label='AE val loss')
        ax2.set_title('Reconstruction Loss')
        ax2.legend()

        ax3.plot([x for x in range(self.train_var.dict_of_vars['CL loss'].epoch)], self.train_var.dict_of_vars['CL loss'].cum_list, label='CL train loss')
        ax3.plot([x for x in range(self.val_var.dict_of_vars['CL loss'].epoch)], self.val_var.dict_of_vars['CL loss'].cum_list, label='CL val loss')
        ax3.set_title('Classification Loss')
        ax3.legend()

        ax4.plot([x for x in range(self.train_var.dict_of_vars['Accuracy'].epoch)], self.train_var.dict_of_vars['Accuracy'].cum_list, label='CL train accuracy')
        ax4.plot([x for x in range(self.val_var.dict_of_vars['Accuracy'].epoch)], self.val_var.dict_of_vars['Accuracy'].cum_list, label='CL val accuracy')
        ax4.set_title('Accuracy (%)')
        ax4.legend()

        if reg:
            ax5.plot([x for x in range(self.train_var.dict_of_vars['Regulariser'].epoch)], self.train_var.dict_of_vars['Regulariser'].cum_list, label='Regularizer (train)')
            ax5.set_title('Regulariser')
            ax5.legend()

        if compare:
            ax5.plot([x for x in range(self.train_var.dict_of_vars['Hessian RSS'].epoch)], self.train_var.dict_of_vars['Regulariser'].cum_list, label='Regularizer (train)')
            ax5.set_title('Hessian RSS')
            ax5.legend()
        os.makedirs(f'./results/{filename}', exist_ok=True)
        plt.savefig(os.path.join(f'./results/{filename}/loss.png'))

