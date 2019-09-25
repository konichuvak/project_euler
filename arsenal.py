import itertools
import math
from types import GeneratorType

import numpy as np


def divisors(n):
    """ Returns all the divisors of a number """
    divs = {1, n}
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.update((i, n // i))
    return divs


Tee = itertools.tee([], 1)[0].__class__


def memo(generator: bool = False):
    """ Memoization helper """
    
    def decorator(f):
        mem = {}
        if generator:
            def wrapper(*args):
                if args not in mem:
                    mem[args] = f(*args)
                if isinstance(mem[args], (GeneratorType, Tee)):
                    # the original can't be used any more,
                    # so we need to change the cache as well
                    mem[args], r = itertools.tee(mem[args])
                    return r
                return mem[args]
        else:
            def wrapper(*args):
                if args not in mem:
                    mem[args] = f(*args)
                return mem[args]
        return wrapper
    
    return decorator


def nCr(n, r):
    """ n Choose r, i.e. number of combinations """
    f = math.factorial
    return int(f(n) / f(r) / f(n - r))


def collection_permutation(nlist, m):
    if m == 0:
        yield []
        return
    
    for list_i in nlist:
        temp = list(nlist)
        temp.remove(list_i)
        for p in collection_permutation(temp, m - 1):
            yield [list_i] + p


def string_permutations(string, output_list, step=0):
    # if we've gotten to the end, print the permutation
    if step == len(string):
        output_list.append(''.join(string))
    
    for i in range(step, len(string)):
        string_copy = [character for character in
                       string]  # copy the string (store as array)
        string_copy[step], string_copy[i] = string_copy[i], string_copy[
            step]  # swap the current index with the step
        string_permutations(string_copy, output_list,
                            step + 1)  # recurse on the portion of the string that has not been swapped yet


def is_prime(n):
    if n == 1:
        return False
    if n % 2 == 0 and n > 2:
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))


def primesfrom2to(n):
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool)
    for i in range(1, int(n ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3::2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


@memo(generator=True)
def partitions(n, I=1):
    """ https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning """
    yield (n,)
    for i in range(I, n // 2 + 1):
        for p in partitions(n - i, i):
            yield (i,) + p


@memo()
def part(n):
    """ https://stackoverflow.com/questions/34421813/recursion-formula-for-integer-partitions """
    p = 0
    if n == 0:
        p += 1
    else:
        k = 1
        while (n >= (k * (3 * k - 1) // 2)) or (n >= (k * (3 * k + 1) // 2)):
            i = (k * (3 * k - 1) // 2)
            j = (k * (3 * k + 1) // 2)
            if (n - i) >= 0:
                p -= ((-1) ** k) * part(n - i)
            if (n - j) >= 0:
                p -= ((-1) ** k) * part(n - j)
            k += 1
    return p
