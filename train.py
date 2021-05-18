import getopt, sys
from stock_predictor import Predictor

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 's:k:', ['symbols=','key='])
    except getopt.GetoptError as err:
        print(err) 
        sys.exit()

    symbols = []
    pkey = ""

    for o, a in opts:
        if o in ('-s', '--symbols') and type(a)==array(str):
            symbols = a
        elif o in ('-k', '--key') and type(a)==str:
            pkey = a 
        else:
            assert False, 'unhandled option'
        
    prd = Predictor(symbol=symbols, key=pkey)
    prd.fetch_dataset()
    prd.arrange_dataset()
    prd.create_model()
    prd.callback()
    prd.train()
    
if __name__ == "__main__":
    main()