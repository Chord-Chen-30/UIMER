# Content should be in this folder

In method UIMER-Dm, a trained model( $\theta$ ) is required to finish the training of our proposed model:

1. Train the extractor ( $\phi$ ) with the model( $\theta$ ) fixed.
2. Train the model( $\theta$ ) with the extractor( $\phi$ ) fixed.

In this folder, a model is required to be trained and the parameters need to be saved to *.state_dict style.
e.g.

```python
torch.save(model.state_dict(), PATH)
```
