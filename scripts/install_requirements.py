import os
from multiprocessing import Process


def pip_install(package):
    cmd = 'pip install {}'.format(package)
    print(cmd)
    try:
        os.system(cmd)
        return 0
    except:
        print('###########################')
        print('Exception')
        print('###########################')
        pip_install(package)


def txt2list(txt_path):
    '''
    Read a txt file and return a list
    Arguments:
        txt_path (str): path to save the text file
    '''
    output = []
    f = open(txt_path, 'r')
    for line in f.readlines():
        output.append(line.replace('\n', ''))
    f.close()
    print('{} loaded.'.format(txt_path))
    return output


def install_once():
    packages = txt2list('./requirements.txt')
    procs = []

    # instantiating process with arguments
    for package in packages:
        proc = Process(target=pip_install, args=(package,))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()


if __name__ == "__main__":  # confirms that the code is under main function
    for i in range(100):
        install_once()
