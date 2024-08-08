# hydra_lightning_template
Base template using hydra/lightning for deep learning.
This is very loosely based on HyenaDNA, but has diverged quite a bit.


### Overview
The idea of this is to create a template for deep learning projects using Hydra and Pytorch Lightning.
It assumes a certain structure for the project, which is the following:

- DataModules
    - Where data logic belongs. This can be simulated or real dataset data. 
    - The existing files are helpers and might not necessarily be needed. Custom datamodules can easily be integrated.
    - The existing simulated allows for custom samplers for different kinds of simulation.
    - The existing real requires a datagetter function, and kind of assumes data is a list of files. This can easilty be replaced.

- Models
    - Where models belong - pytorch models that know nothing about lightning. 

- Task
    - A task is kind of like a decoder head. We could have the entirely same base model, however, we might be able to represent the output in different ways- for example if we wanted an integer output then maybe we would have a task where the loss/decoder pair treats ouputs continuously and another that treats them as categories. 
    - This logic is not defined on the decoder, because you might want the same decoder architecture for multiple tasks. The decoder class is a static attribute of the task though.
    - This logic is also decoupled from the datamodule, because you might want to train the same task on different data, or different tasks on the same data.


### Other notes
- I have included vscode launch.json for debugging, I also include the .env file. Both of these things allow me to treat it like a package without pip installing for now.
- I do import schemas.py for registering the default schema values. This is not necessary, and without this the src code can be treated as project root. It also depends how once likes to do/setup tests. At the moment src is not the project root, hence imports reflect this.
- You can optionally use git hooks. I find them annoying sometimes but they also make code formatting easier.
