import joblib

if __name__ == '__main__':
    res = joblib.load("loss_data.data")

    # x = [i for i in range(len(res))]
    x = [0.949068108108108, 0.9513381780293331, 0.9432756509885302, 0.9449964328594547, 0.9295989169064139,
         0.9321580516138893, 0.9332122074338164, 0.9380941259914715, 0.9228478314514003, 0.9205009626977129]
    x1 = [0.9429583611439133, 0.939971049373027, 0.943438953437734, 0.9387836872637318, 0.93590897062715,
          0.9346181150762732, 0.929482106744043, 0.9335735313502351, 0.928703986059682, 0.9234423146537643]

    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(range(1, 11), x)
    plt.plot(range(1, 11), x1)
    plt.title("F1-epoch")

    plt.xlabel("epoch")
    plt.ylabel("F1")
    plt.legend(['My method', 'Baseline'])
    plt.savefig('image.png')
    plt.show()
