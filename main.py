import utils
from train import Trainer

checkpoint = dict()
parser = utils.set_up_args()
args = parser.parse_args()
trainer = Trainer(checkpoint, args)
trainer.run()
trainer.save_checkpoint('end')