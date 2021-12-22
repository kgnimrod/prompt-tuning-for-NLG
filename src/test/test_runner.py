import unittest


def load_suite():
    suite = unittest.defaultTestLoader.discover(start_dir='.', pattern='test_*.py', top_level_dir=None)
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(load_suite())
