import torch

def main():
    params = torch.load(r"./Models/trained_models/DSS_ZINC__ZINC_val_0.1117_test_0.0909_13:30:36.pt")
    print(len(params))

    print(list(filter(lambda key: "final" in key, params.keys())))

if __name__ == "__main__":
    main()