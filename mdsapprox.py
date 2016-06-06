import unittest

import click


@click.group()
def main():
    click.echo('Multidimensional scaling approximations.')


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

if __name__ == '__main__':
    main()
