import numpy as np
import matplotlib.pyplot as plt

class SteepestDescent:
    def __init__(self, particles, n_particles, box_length, tol=1e-12, max_iter=100000, min_particle_dist=1e-8):
        self.n = n_particles
        self.L = box_length
        self.tol = tol
        self.max_iter = max_iter
        self.Eplison = min_particle_dist
        self.Energies = []
        self.positions = particles

    def potential_energy(self, positions=None):
        if positions is None:
            positions = self.positions
        U = 0.0
        for i in range(self.n):
            for j in range(i+1, self.n):
                rij = np.linalg.norm(positions[i] - positions[j])
                U += -1.0 / (rij + self.Eplison)
        return U

    def gradient(self, positions=None):
        if positions is None:
            positions = self.positions
        grad = np.zeros_like(positions)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    rij_vec = positions[i] - positions[j]
                    rij = np.linalg.norm(rij_vec)
                    if rij > 1e-8:
                        grad[i] += rij_vec / ((rij + self.Eplison)**3)
        return grad

    def line_search(self, direction, grad):
        alpha = 1.0  
        beta = 0.5  
        c = 1e-4    

        current_energy = self.potential_energy()
        while alpha > 1e-8:
            new_positions = self.positions + alpha * direction
            new_positions = np.clip(new_positions, -self.L/2, self.L/2)
            new_energy = self.potential_energy(new_positions)

            # Armijo condition
            if new_energy <= current_energy + c * alpha * np.sum(grad * direction):
                return alpha
            alpha *= beta
        return alpha

    def minimize(self):
        prev_energy = self.potential_energy()

        if self.Energies == []:
            self.Energies.append(prev_energy)

        for step in range(self.max_iter):
            grad = self.gradient()
            direction = -grad

            if np.linalg.norm(grad)<=self.tol:
                print(f"Converged at step {step}, Energy = {self.Energies[-1]:.6f}")
                break

            alpha = self.line_search(direction, grad)
            self.positions += alpha * direction
            self.positions = np.clip(self.positions, -self.L/2, self.L/2)

            energy = self.potential_energy()
            self.Energies.append(energy)
            prev_energy = energy

        return self.positions, self.Energies[-1]

    def energy_plot(self):
        plt.figure(figsize=(6,4))
        plt.plot(self.Energies)
        plt.xlabel("Iteration")
        plt.ylabel("Potential Energy")
        plt.title("Energy Minimization")
        plt.grid(True)
        plt.show()

    def final_positions_plot(self):
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.positions[:,0], self.positions[:,1], self.positions[:,2], c='red', s=50)
        ax.set_xlim([-self.L/2, self.L/2])
        ax.set_ylim([-self.L/2, self.L/2])
        ax.set_zlim([-self.L/2, self.L/2])
        ax.set_title("Final Particle Positions")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

class ConjugateGradient:
    def __init__(self, particles, n_particles, box_length, tol=1e-12, max_iter=100000, min_particle_dist=1e-8):
        self.n = n_particles
        self.L = box_length
        self.tol = tol
        self.max_iter = max_iter
        self.Eplison = min_particle_dist
        self.Energies = []
        self.positions = particles

    def potential_energy(self, positions=None):
        if positions is None:
            positions = self.positions
        U = 0.0
        for i in range(self.n):
            for j in range(i+1, self.n):
                rij = np.linalg.norm(positions[i] - positions[j])
                U += -1.0 / (rij + self.Eplison)
        return U

    def gradient(self, positions=None):
        if positions is None:
            positions = self.positions
        grad = np.zeros_like(positions)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    rij_vec = positions[i] - positions[j]
                    rij = np.linalg.norm(rij_vec)
                    if rij > 1e-8:
                        grad[i] += rij_vec / ((rij + self.Eplison)**3)
        return grad

    def line_search(self, direction, grad):
        alpha = 1.0  
        beta = 0.5   
        c = 1e-4  

        current_energy = self.potential_energy()
        while alpha > 1e-8:
            new_positions = self.positions + alpha * direction
            new_positions = np.clip(new_positions, -self.L/2, self.L/2)
            new_energy = self.potential_energy(new_positions)

            # Armijo condition
            if new_energy <= current_energy + c * alpha * np.sum(grad * direction):
                return alpha
            alpha *= beta
        return alpha

    def minimize(self):
        grad = self.gradient()
        direction = -grad
        prev_energy = self.potential_energy()
        self.Energies.append(prev_energy)

        for step in range(self.max_iter):
            if np.linalg.norm(grad)<=self.tol:
                print(f"Converged at step {step}, Energy = {self.Energies[-1]:.6f}")
                break

            alpha = self.line_search(direction, grad)
            self.positions += alpha * direction
            self.positions = np.clip(self.positions, -self.L/2, self.L/2)

            new_grad = self.gradient()
            energy = self.potential_energy()
            self.Energies.append(energy)

            # Fletcher-Reeves beta update
            beta = np.sum(new_grad*new_grad) / (np.sum(grad*grad) + 1e-12)
            direction = -new_grad + beta * direction

            grad = new_grad
            prev_energy = energy

        return self.positions, self.Energies[-1]

    def energy_plot(self):
        plt.figure(figsize=(6,4))
        plt.plot(self.Energies)
        plt.xlabel("Iteration")
        plt.ylabel("Potential Energy")
        plt.title("Energy Minimization (Conjugate Gradient)")
        plt.grid(True)
        plt.show()

    def final_positions_plot(self):
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.positions[:,0], self.positions[:,1], self.positions[:,2], c='blue', s=50)
        ax.set_xlim([-self.L/2, self.L/2])
        ax.set_ylim([-self.L/2, self.L/2])
        ax.set_zlim([-self.L/2, self.L/2])
        ax.set_title("Final Particle Positions (CG)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()


# Example usage:
if __name__ == "__main__":
    particles = np.random.uniform(-5, 5, (10,3))

    sd = SteepestDescent(particles, n_particles=10, box_length=10)
    final_positions, final_energy = sd.minimize()
    print("Final positions SD:\n", final_positions)
    print("Final energy:", final_energy)
    sd.energy_plot()
    sd.final_positions_plot()

    cg = ConjugateGradient(particles, n_particles=10, box_length=10)
    final_positions, final_energy = cg.minimize()
    print("Final positions CG:\n", final_positions)
    print("Final energy:", final_energy)
    cg.energy_plot()
    cg.final_positions_plot()
