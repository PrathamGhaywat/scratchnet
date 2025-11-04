![Hackatime Badge](https://hackatime-badge.hackclub.com/U097N0AKR6Z/scratchnet)

<div align="center">
  <a href="https://moonshot.hackclub.com" target="_blank">
    <img src="https://hc-cdn.hel1.your-objectstorage.com/s/v3/35ad2be8c916670f3e1ac63c1df04d76a4b337d1_moonshot.png" 
         alt="This project is part of Moonshot, a 4-day hackathon in Florida visiting Kennedy Space Center and Universal Studios!" 
         style="width: 100%;">
  </a>
</div>

# ScratchNet - A Neural Network built from Scratch

This is a Neural Network built from Scratch using Python and NumPy. 
Try it out: https://scratchnet-mnist.streamlit.app/

Dataset: MNIST (70000 handwritten digits)
Architecture: [784 -> 128 -> 64 ->]
Final Accuracy: 97.61%

## How to use

To run this please follow the set of instructions below.

REQUIREMENTS:
1. Python 3.13
2. Pip
3. Git (optional)

First clone the Repository (Alternatively you can download the ZIP file via clicking on the "<> Code" button and then navigating to the "Local" Tab and clicking on the "Download ZIP" option. Then you can extract it onto your Desktop):
```bash
git clone https://github.com/PrathamGhaywat/scratchnet.git
```

OPTIONAL: Creating an Virtual Enviroment will help you seprate the libraries from your main python compiler and mitigates the risk of corrupting it:

```bash
python -m venv scratchnet

#Activate the venv: on Linux and MacOS it would be different: source scratchnet/bin/activate
./scratchnet/scripts/activate
```
Then run: 
```bash
pip install -r requirements.txt
```

Then run train_mnist.py
```bash
python src/core/train_mnist.py
```

This will train the neural network. But you can skip that process if you want and directly run the Web GUI:
```bash
streamlit run src/main.py
```

Then you can visit the given localhost address and use it!
## License

The license is Apache License 2.0 - For more info see: [License](LICENSE.md)

