from stock_predictor import Predictor
from train import symbols, pkey

def main():
    prd = Predictor(symbol=symbols, key=pkey)
    print(symbols, prd.predict())
    
if __name__ == "__main__":
    main()