from typing import Tuple, List

def read_off(file_path: str):
    """
    reading files in the OFF format
    :param file_path: (str) Path to the file
    :return: ((list, list)) --> ((list of vs X, Y, Z,  indices of all vs for each face))
    """

    with open(file_path, 'r') as file:

        first_line = file.readline().strip()
        assert first_line == 'OFF', "not a valid OFF header"

        n_vs, n_fc, _ = [int(string)for string in file.readline().strip().split()]

        vs = [[float(vertice) for vertice in file.readline().strip().split()] for _ in range(n_vs)]

        fc = [[int(face) for face in file.readline().strip().split()[1:]] for _ in range(n_fc)]

    return vs, fc

def write_off(data: Tuple[List, List], save_path: str):
    """
    writing files in the OFF format

    :param data: ((list, list)) -->  (list of vs X, Y, Z , list contains the indices of all vs for each face)
    :param save_path: (str) Path to save the file to
    (ending with .off)
    """

    assert save_path.endswith('.off'), "'save_path' should end with '.off'"

    vs, fc = data
    n_vs = len(vs)
    n_fc = len(fc)

    with open(save_path, 'w') as file:
        # Write header
        file.write('OFF\n')

        file.write(f'{n_vs} {n_fc} 0\n')

        [file.write(f"{vertice[0]} {vertice[1]} {vertice[2]}\n")for vertice in vs]

        [file.write(f"{len(face)} {' '.join([str(f) for f in face])}\n")for face in fc]

