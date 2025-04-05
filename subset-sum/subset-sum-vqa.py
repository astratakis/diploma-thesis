from qiskit import *
import numpy as np

from scipy.optimize import minimize

import matplotlib.pyplot as plt


EXPERIMENTS = 50
cost_evolution = []
experiment_results = []

def vqe(theta: list) -> QuantumCircuit:

    qc = QuantumCircuit(N)

    for _ in range(N):
        qc.h(_)

    for irep in range(layers):
        
        for i in range(N):

            qc.ry(theta[irep * N + i], i)

        # Next Apply layers of entangling gates...

        for i in range(0, N, 2):
            if i+1 < N:
                qc.cx(i, i+1)
        for i in range(1, N, 2):
            if i+1 < N:
                qc.cx(i, i+1)

    qc.measure_all()
    return qc

def compute_expectation(counts: dict) -> float:

    total_counts = 0
    sum = 0

    for key in counts.keys():

        value = counts[key]

        inner_sum = 0

        for i in range(N):
            if key[::-1][i] == '1':
                inner_sum += a[i]

        sum += ((S - inner_sum)**2) * value

        total_counts += value

    avg = sum / total_counts

    cost_evolution.append(avg)

    return avg

def get_expectation():

    backend = Aer.get_backend('qasm_simulator')
    
    def execute_circ(theta):
        
        qc = vqe(theta)
        counts = backend.run(qc, shots=10000).result().get_counts()
        
        return compute_expectation(counts)
    
    return execute_circ


MAX_ITER = 500

counts_list = []

np.random.seed(1823746)

diffs = dict()

for exp in range(EXPERIMENTS):
    a = np.random.randint(1, 20, size=15)
    S = np.sum(np.random.choice(a, size=4))

    print(a)
    print('Sum:', S)

    N = len(a)
    layers = 2

    cost_evolution = list()
    print('Running experiment:', (exp+1))
    expectation = get_expectation()

    theta = np.random.random(layers * N)

    res = minimize(expectation, theta, method='COBYLA', options={"maxiter":MAX_ITER})

    experiment_results.append(cost_evolution + [cost_evolution[-1]] * (MAX_ITER - len(cost_evolution)))


    final_circuit = vqe(res.x)
    backend = Aer.get_backend('qasm_simulator')
    counts = backend.run(final_circuit, shots=100).result().get_counts()

    print(counts)

    for key in counts:

        temp_sum = 0

        value = counts[key]

        for i in range(N-1, -1, -1):
            if key[i] == '1':
                temp_sum += a[i]

        difference = np.abs(S - temp_sum)

        if difference not in diffs.keys():
            diffs[difference] = counts[key]
        else:
            diffs[difference] += counts[key]


result = np.zeros((len(diffs.keys()), 2), dtype=int)

keys = list(diffs.keys())

for index, key in enumerate(keys):
    result[index][0] = key
    result[index][1] = diffs[key]

print(diffs)
print(result)

np.save("vqa-susbset-sum.npy", result)

experiment_results = np.array(experiment_results)

upper_bound = [np.max(experiment_results[:, i]) for i in range(MAX_ITER)]
lower_bound = [np.min(experiment_results[:, i]) for i in range(MAX_ITER)]
mean = [np.mean(experiment_results[:, i]) for i in range(MAX_ITER)]

np.savez("data.npz", upper_bound=upper_bound, lower_bound=lower_bound, mean=mean)

data = np.load("data.npz")

upper_bound = data['upper_bound']
lower_bound = data['lower_bound']
mean = data['mean']

plt.figure()
plt.plot(range(MAX_ITER), mean, label='VQA', color='blue')
plt.fill_between(range(MAX_ITER), lower_bound, upper_bound, color='lightblue')
plt.plot(range(MAX_ITER), [0] * MAX_ITER, label='Best Solution')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Evolution')
plt.show()
