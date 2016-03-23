import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser('output script for integration test')
    parser.add_argument('first_file', type=str,
                        help='File containing "First,<factor>')
    parser.add_argument('second_file', type=str,
                        help='File containing "Second,<factor>')
    parser.add_argument('-o', type=str, help='output filename', default='')

    args = parser.parse_args()

    with open(args.first_file) as f:
        contents = f.read().strip()
        first_factor = float(contents.split(',')[1])
    with open(args.second_file) as f:
        contents = f.read().strip()
        second_factor = float(contents.split(',')[1])

    result = first_factor + second_factor
    results_string = 'ResponseA,{0}\n'.format(result)
    out_name = 'results.csv' if not args.o else args.o

    with open(out_name, 'w') as f:
        f.write(results_string)