from qiskit import *
import numpy as np

from scipy.optimize import minimize

import matplotlib.pyplot as plt


def qaoa_circuit(theta: list) -> QuantumCircuit:

    qc = QuantumCircuit(N)

    p = len(theta)//2

    gamma = theta[:p]
    beta = theta[p:]

    for _ in range(N):
        qc.h(_)

    for irep in range(p):
        
        # First apply the ising Hamiltonian...

        qc.barrier()

        for i in range(N-1):
            for j in range(i+1, N):
                qc.rzz(gamma[irep] * a[i] * a[j] / 2, i, j)

        qc.barrier()

        for i in range(N):
            qc.rz(-gamma[irep] * L *a[i], i)

        # Next apply the mixing hamiltonian...

        for i in range(N):
            qc.rx(2 * beta[irep], i)

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
        
        qc = qaoa_circuit(theta)
        counts = backend.run(qc, shots=10000).result().get_counts()
        
        return compute_expectation(counts)
    
    return execute_circ

MAX_ITER = 400

layer_experiments = [1, 10, 20]

layer_experimental_results = []
layer_counts = []

for layers in layer_experiments:

    experimental_measurements.clear()
    counts_measurements = dict()

    print('Number of layers:', layers)

    for iter in range(EXPERIMENTS):

        cost_evolution.clear()
        theta = np.random.random(2*layers)

        print('Running experiment:', (iter + 1))

        expectation = get_expectation()
        res = minimize(expectation, theta, method='COBYLA', options={"maxiter":MAX_ITER})

        experimental_measurements.append(cost_evolution + [cost_evolution[-1]] * (MAX_ITER - len(cost_evolution)))

        final_circuit = qaoa_circuit(res.x)
        backend = Aer.get_backend('qasm_simulator')
        counts = backend.run(final_circuit, shots=10000).result().get_counts()

        for key in counts.keys():
            if key[::-1] not in counts_measurements.keys():
                counts_measurements[key[::-1]] = 0
                counts_measurements[key[::-1]] += counts[key]
            else:
                counts_measurements[key[::-1]] += counts[key]

    layer_experimental_results.append(experimental_measurements.copy())
    layer_counts.append(counts_measurements)

TOTAL_COUNTS = 10000 * EXPERIMENTS

strings = list()

for i in range(8):
    binary_str = format(i, '03b')  # '03b' formats the number as a 3-digit binary string
    strings.append(binary_str)


plt.figure()
plt.subplot(3, 2, 1)

experimental_measurements = np.array(layer_experimental_results[0])

upper_bound = [np.max(experimental_measurements[:, i]) for i in range(MAX_ITER)]
lower_bound = [np.min(experimental_measurements[:, i]) for i in range(MAX_ITER)]
mean = [np.mean(experimental_measurements[:, i]) for i in range(MAX_ITER)]

plt.plot(range(MAX_ITER), mean, label='QAOA (p=1)', color='blue')
plt.fill_between(range(MAX_ITER), lower_bound, upper_bound, color='lightblue')
plt.legend()
plt.ylabel('Cost')

plt.subplot(3, 2, 2)

prob = list()
for key in strings:
    if key not in layer_counts[0].keys():
        prob.append(int(0))
    else:
        prob.append(100 * layer_counts[0][key]/TOTAL_COUNTS)

plt.bar(strings, prob, color=['#1f77b4', 'green', 'green', '#1f77b4', 'green', '#1f77b4', '#1f77b4', '#1f77b4'])
plt.xlabel('Output bitstring')
plt.ylabel('Probability (%)')

plt.subplot(3, 2, 3)
experimental_measurements2 = np.array(layer_experimental_results[1])

upper_bound2 = [np.max(experimental_measurements2[:, i]) for i in range(MAX_ITER)]
lower_bound2 = [np.min(experimental_measurements2[:, i]) for i in range(MAX_ITER)]
mean2 = [np.mean(experimental_measurements2[:, i]) for i in range(MAX_ITER)]

plt.plot(range(MAX_ITER), mean2, label='QAOA (p=10)', color='blue')
plt.fill_between(range(MAX_ITER), lower_bound2, upper_bound2, color='lightblue')
plt.legend()
plt.ylabel('Cost')

plt.subplot(3, 2, 4)

prob = list()
for key in strings:
    if key not in layer_counts[1].keys():
        prob.append(int(0))
    else:
        prob.append(100 * layer_counts[1][key]/TOTAL_COUNTS)

plt.bar(strings, prob, color=['#1f77b4', 'green', 'green', '#1f77b4', 'green', '#1f77b4', '#1f77b4', '#1f77b4'])
plt.xlabel('Output bitstring')
plt.ylabel('Probability (%)')

plt.subplot(3, 2, 5)
experimental_measurements3 = np.array(layer_experimental_results[2])

upper_bound3 = [np.max(experimental_measurements3[:, i]) for i in range(MAX_ITER)]
lower_bound3 = [np.min(experimental_measurements3[:, i]) for i in range(MAX_ITER)]
mean3 = [np.mean(experimental_measurements3[:, i]) for i in range(MAX_ITER)]

plt.plot(range(MAX_ITER), mean3, label='QAOA (p=20)', color='blue')
plt.fill_between(range(MAX_ITER), lower_bound3, upper_bound3, color='lightblue')
plt.legend()
plt.ylabel('Cost')

plt.subplot(3, 2, 6)
prob = list()
for key in strings:
    if key not in layer_counts[2].keys():
        prob.append(int(0))
    else:
        prob.append(100 * layer_counts[2][key]/TOTAL_COUNTS)

plt.bar(strings, prob, color=['#1f77b4', 'green', 'green', '#1f77b4', 'green', '#1f77b4', '#1f77b4', '#1f77b4'])
plt.xlabel('Output bitstring')
plt.ylabel('Probability (%)')

plt.show()
