import argparse
import os
import models
import torch
import train_controller

# Create the parser
parser = argparse.ArgumentParser(description='TUI parser')

# Add arguments
parser.add_argument('TMN', help='Trained model name')
parser.add_argument('mode', help='Wanted mode', choices=['train', 'test'])
parser.add_argument('--training_type', choices=['base', 'tuneup', 'wo_syn', 'wo_cur'], default='base', help='Training type')
parser.add_argument('--test_type', choices=['transductive', 'inductive'], default='transductive', help='Test type')
parser.add_argument('--epoch', type=int, default=1000, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
parser.add_argument('--drop_percent', type=float, default=0.5, help='Drop percent')

# Parse the command-line arguments
args = parser.parse_args()

# Access the values
print('TMN:', args.TMN)
print('mode:', args.mode)
print('training_type:', args.training_type)
print('test_type:', args.test_type)
print('epoch:', args.epoch)
print('lr:', args.lr)
print('weight_decay:', args.weight_decay)
print('drop_percent:', args.drop_percent)



trainer = train_controller.train_controller()

# Set hyperparameters

optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# Does model exists in models folder?
model = models.GCN(128)
model_exists = False
models_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
model_path = os.path.join(models_folder, args.TMN)
if os.path.exists(model_path):
    model_exists = True
    model.load_state_dict(torch.load(model_path))

else:
    trainer.tuneup(mode=args.training_type,epochs=args.epoch,drop_percent=args.drop_percent, optimizer=optimizer)

    



print("Setting up trainer... This could take a while (~10 minutes)")
trainer.model = model

if args.mode == 'train':
    print('Training...')
    trainer.tuneup(mode=args.training_type,epochs=args.epoch,drop_percent=args.drop_percent, optimizer=optimizer)

elif args.mode == 'test':
    print('Testing...')
    if args.test_type == 'transductive':
        
        recall,_ = trainer.test_transductive()
        
    elif args.test_type == 'inductive':
        recall,_ =trainer.test_inductive()
    print('Recall:', recall)

