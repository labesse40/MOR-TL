import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI


class MpiDecomposition:

    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()  # Rang (identifiant) du processus
        self.size = comm.Get_size()  # Nombre total de processus

        self.proc_x, self.proc_y = self.decompose_domain_mpi()
        self.pos_x = self.rank % self.proc_x
        self.pos_y = self.rank // self.proc_x

        self.neighbors, self.ghost_layers_left_right, self.ghost_layers_up_down = self.get_neighbors()


    def decompose_domain_mpi(self):
        size = self.size
        factors = [(i, size // i) for i in range(1, size + 1) if size % i == 0]
        best_factor = min(factors, key=lambda x: abs(x[0] - x[1]))
        return best_factor


    def get_neighbors(self):
        neighbors = {}
        ghost_layers_left_right = 0
        ghost_layers_up_down = 0
        pos_x = self.pos_x
        pos_y = self.pos_y
        rank = self.rank
        proc_x = self.proc_x
        proc_y = self.proc_y

        if pos_y > 0:
            neighbors['left'] = rank - proc_x
            ghost_layers_left_right += 1
        else:
            neighbors['left'] = MPI.PROC_NULL  # Pas de voisin à gauche
        if pos_y < proc_y - 1:
            neighbors['right'] = rank + proc_x
            ghost_layers_left_right += 1
        else:
            neighbors['right'] = MPI.PROC_NULL  # Pas de voisin à droite
        if pos_x > 0:
            neighbors['up'] = rank - 1
            ghost_layers_up_down += 1
        else:
            neighbors['up'] = MPI.PROC_NULL  # Pas de voisin en bas
        if pos_x < proc_x - 1:
            neighbors['down'] = rank + 1
            ghost_layers_up_down += 1
        else:
            neighbors['down'] = MPI.PROC_NULL  # Pas de voisin en haut

        if pos_y > 0 and pos_x > 0:
            neighbors['up-left'] = rank - proc_x - 1
        else:
            neighbors['up-left'] = MPI.PROC_NULL  # Pas de voisin à gauche
        if pos_y < proc_y - 1 and pos_x > 0:
            neighbors['up-right'] = rank + proc_x - 1
        else:
            neighbors['up-right'] = MPI.PROC_NULL  # Pas de voisin à droite
        if pos_x < proc_x - 1 and pos_y > 0 :
            neighbors['down-left'] = rank - proc_x + 1
        else:
            neighbors['down-left'] = MPI.PROC_NULL  # Pas de voisin en bas
        if pos_x < proc_x - 1 and pos_y < proc_y - 1:
            neighbors['down-right'] = rank + proc_x + 1
        else:
            neighbors['down-right'] = MPI.PROC_NULL  # Pas de voisin en haut

        return neighbors, ghost_layers_left_right, ghost_layers_up_down


    def synchronize(self, grid, subdomain):
        comm = self.comm
        neighbors = self.neighbors
        # On utilise les buffers temporaires pour envoyer et recevoir

        gn_l = subdomain.ghost_nodes['left']
        gn_r = subdomain.ghost_nodes['right']
        gn_u = subdomain.ghost_nodes['up']
        gn_d = subdomain.ghost_nodes['down']
        gn_ul = subdomain.ghost_nodes['up-left']
        gn_ur = subdomain.ghost_nodes['up-right']
        gn_dl = subdomain.ghost_nodes['down-left']
        gn_dr = subdomain.ghost_nodes['down-right']

        send_l = subdomain.send_nodes['left']
        send_r = subdomain.send_nodes['right']
        send_u = subdomain.send_nodes['up']
        send_d = subdomain.send_nodes['down']
        send_ul = subdomain.send_nodes['up-left']
        send_ur = subdomain.send_nodes['up-right']
        send_dl = subdomain.send_nodes['down-left']
        send_dr = subdomain.send_nodes['down-right']

        # Préparer les communications avec les voisins
        for direction, neighbor in neighbors.items():
            if neighbor != MPI.PROC_NULL:
                if direction == 'left':
                    send = grid[send_l]
                    comm.isend(send, dest=neighbor, tag=0)
                    req = comm.irecv(source=neighbor, tag=1)
                    grid[gn_l] = req.wait()

                elif direction == 'right':
                    send = grid[send_r]
                    comm.isend(send, dest=neighbor, tag=1)
                    req = comm.irecv(source=neighbor, tag=0)
                    grid[gn_r] = req.wait()

                elif direction == 'up':
                    send = grid[send_u]
                    comm.isend(send, dest=neighbor, tag=2)
                    req = comm.irecv(source=neighbor, tag=3)
                    grid[gn_u] = req.wait()

                elif direction == 'down':
                    send = grid[send_d]
                    comm.isend(send, dest=neighbor, tag=3)
                    req = comm.irecv(source=neighbor, tag=2)
                    grid[gn_d] = req.wait()

                elif direction == 'up-left':
                    send = grid[send_ul]
                    comm.isend(send, dest=neighbor, tag=4)
                    req = comm.irecv(source=neighbor, tag=5)
                    grid[gn_ul] = req.wait()

                elif direction == 'up-right':
                    send = grid[send_ur]
                    comm.isend(send, dest=neighbor, tag=6)
                    req = comm.irecv(source=neighbor, tag=7)
                    grid[gn_ur] = req.wait()

                elif direction == 'down-left':
                    send = grid[send_dl]
                    comm.isend(send, dest=neighbor, tag=7)
                    req = comm.irecv(source=neighbor, tag=6)
                    grid[gn_dl] = req.wait()

                elif direction == 'down-right':
                    send = grid[send_dr]
                    comm.isend(send, dest=neighbor, tag=5)
                    req = comm.irecv(source=neighbor, tag=4)
                    grid[gn_dr] = req.wait()

        comm.Barrier()

        return grid
