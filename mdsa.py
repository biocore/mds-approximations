import os
import unittest
from pprint import pprint, pformat

import click
import numpy as np

from mdsa.algorithm import Algorithm
from mdsa.algorithms.nystrom import Nystrom
from mdsa.algorithms.scmds import Scmds

# Register algorithms here
Algorithm.register(Nystrom())
Algorithm.register(Scmds())

@click.group()
def main():
    click.echo('Multidimensional scaling approximations. For command line options, re-run with --help flag.')


@click.command()
@click.option('--verbosity', type=click.Choice(['quiet', 'default', 'verbose']), default='verbose',
              help='Python unittest package: level of detail to display in output')
@click.option('--directory', type=click.Path(), default='.',
              help='Relative path of directory containing tests')
def test(verbosity, directory):
    click.echo('Running tests via unittest from directory (output verbosity level: %s): %s' % (verbosity, directory))
    verbosity_codes = dict(quiet=0, default=1, verbose=2)

    # Run all tests
    testsuite = unittest.TestLoader().discover(directory)
    unittest.TextTestRunner(verbosity=verbosity_codes[verbosity]).run(testsuite)


main.add_command(test)


@click.command()
# inputfile = Path to input .txt file containing a numpy-readable distance matrix
@click.argument('inputfile', type=click.File('r'))
@click.option('--outpath', type=click.Path(), default='./out', help='Directory where we should output .txt files.')
@click.option('--algorithm', 'algorithms', type=str, default=['all'], multiple=True,
              help='Algorithm to run. Pass multiple '
                   '`--algorithm alg_name_here` flags to run '
                   'more than one. Omit entirely to run all.\n'
                   'Algorithms: %s' % (', '.join(Algorithm.algorithms.keys())))
def run(inputfile, outpath, algorithms):
    click.echo('Running these algorithms on given matrix (from `%s` and outputting to `%s`): %s\n'
               % (inputfile.name, outpath, ', '.join(algorithms)))

    # TODO: sanity-check the format of this file and document supported input formats
    input_matrix = np.loadtxt(inputfile.name)

    if len(algorithms) == 1 and algorithms[0] == 'all':
        algorithms = Algorithm.algorithms.keys()

    # Run algorithms
    for algorithm_name in algorithms:
        click.echo('> Running algorithm %s on distance matrix from input file: %s' % (algorithm_name, inputfile.name))
        algorithm = Algorithm.get_algorithm(algorithm_name)
        eigenvectors, eigenvalues, percentages = algorithm.run(input_matrix)

        formatted_pcoa_output = format_pcoa_output(eigenvectors, eigenvalues, percentages)

        outfile_full_path = os.path.join(outpath, algorithm_name + '.txt')

        if not os.path.exists(outpath):
            os.makedirs(outpath)

        click.echo('Outputting results to file: %s\n' % outfile_full_path)
        with open(outfile_full_path, 'w') as out:
            out.write(formatted_pcoa_output)
            out.write('\n')
            out.flush()


def format_pcoa_output(eigenvectors, eigenvalues, percentages):
    eigenvectors = np.array(eigenvectors)
    eigenvalues = np.array(eigenvalues)
    percentages = np.array(percentages)
    str_out = ''

    # Output eigenvalues
    if len(eigenvalues) > 0 and not np.all(np.isnan(eigenvalues)):
        str_out += 'Eigenvalues\t(%d total)\n\n' % len(eigenvalues)
        for eigval in eigenvalues:
            str_out += str(eigval)
            str_out += '\n'
    else:
        str_out += 'No eigenvalues output by algorithm.'
    str_out += '\n\n'

    # Output proportion explained percentages
    if len(percentages) > 0 and not np.all(np.isnan(percentages)):
        str_out += 'Proportion explained as percentages \t(%d total)\n\n' % len(percentages)
        for percentage in eigenvalues:
            str_out += str(percentage)
            str_out += '%\n'
    else:
        str_out += 'No percentages output by algorithm.'

    str_out += '\n\n'

    # Output eigenvectors
    str_out += 'Eigenvectors (matrix shape: %s)\n' % repr(eigenvectors.shape)

    str_out += repr(eigenvectors)
    str_out += '\n'

    return str_out


main.add_command(run)

if __name__ == '__main__':
    main()
