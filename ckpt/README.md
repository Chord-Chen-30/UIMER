# Content should be in this folder

In method UIMER-Dm, a trained model( $\theta$ ) is required to finish the training of our proposed model:

1. Train the extractor ( $\phi$ ) with the model( $\theta$ ) fixed.
2. Train the model( $\theta$ ) with the extractor( $\phi$ ) fixed.

In this folder, a model is required to be trained and the parameters need to be saved to *.state_dict style w.r.t. each task and each random seed.

## Python Command

```python
torch.save(model.state_dict(), PATH)
```

## .state_dict example

- For Intent Classification task
  - __intent10shotinput55.state_dict__ means a trained model on Intent Classification task in 10-shot setting. "input" means the method from [De Cao et al.](https://github.com/nicola-decao/diffmask) and there is another option "hidden". 55 is a random seed.

- For Slot Filling task
  - __slot-1shotinput12333.state_dict__ means a trained model on Slot Filling task in full resource setting. "input" means the method from [De Cao et al.](https://github.com/nicola-decao/diffmask) and there is another option "hidden". 12333 is a random seed.

- For Natural Language Infernece task
  - __66esnlibert.state_dict__ & __66esnliclassifier.state_dict__. There are 2 files for  Natural Language Infernece task: 1. trained bert parameters. 2. trained linear classifier parameters. 66 is a random seed.
  