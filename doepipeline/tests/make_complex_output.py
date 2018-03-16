import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser('output script for integration test')
    parser.add_argument('first_file', type=str,
                        help='File containing "First,<factor>')
    parser.add_argument('second_file', type=str,
                        help='File containing "Second,<factor>')
    parser.add_argument('categorical_factor', type=str,
                        help='Categorical factor', choices=('A', 'B'))
    parser.add_argument('-o', type=str, help='output filename', default='')
    parser.add_argument('-x', type=float, help='optimal x', default=0.0)
    parser.add_argument('-y', type=float, help='optimal y', default=0.0)

    args = parser.parse_args()

    with open(args.first_file) as f:
        contents = f.read().strip()
        first_factor = float(contents.split(',')[1])
    with open(args.second_file) as f:
        contents = f.read().strip()
        second_factor = float(contents.split(',')[1])

    x = first_factor - args.x
    y = second_factor - args.y

    result_1 = (x ** 2 + y ** 2 + x * y) / 1000
    result_2 = ((x - 1) ** 2 + (y - 1) ** 2) / 1000

    if args.categorical_factor == 'A':
        result_2 += .5

    results_string = 'ResponseA,{0}\nResponseB,{1}'.format(result_1, result_2)
    out_name = 'results.csv' if not args.o else args.o

    with open(out_name, 'w') as f:
        f.write(results_string)