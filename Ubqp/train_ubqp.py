import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from instance import getInstance
from simple_UBQP import Encoder_Embedding, DRL4UBQP
from instance import testInstance
from heuri import DRLBISolution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_transition_time(p1, p2):
    transition_time = torch.exp(
        -0.5 * (p1 * p1 + p2 * p2))
    return transition_time

class StateCritic(nn.Module):

    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()

        self.static_encoder = Encoder_Embedding(2, hidden_size)
        self.fc1 = nn.Conv1d(hidden_size * 1, 20, kernel_size=1)  #
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)  # fliter20
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)  # fliter20
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, Q_DIAG, Q_RC):
        static = torch.cat((Q_DIAG, Q_RC), dim=1)[:, :, 1:]
        static_hidden = self.static_encoder(static)
        output = F.relu(self.fc1(static_hidden))  # relu
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output


def validate(data_loader, actor, RLLS, reward_fn, render_fn=None, save_dir='.',seed= 1,num_plot=5):

    actor.eval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    rewards = []
    rewards_LS=[]
    time_ = 0.

    for batch_idx, batch in enumerate(data_loader):
        x0, dynamic, Q = batch  # datasfile":('...','...','...')

        num_samples, input_size,_ = Q.size()
        p0 = torch.zeros(num_samples, 1)
        p1 = torch.zeros(num_samples, input_size, 1)
        p2 = torch.zeros(num_samples, 1, input_size + 1)

        idx = 0
        for i in Q:
            x = i.numpy()
            X = np.triu(x)
            X += X.T - np.diag(X.diagonal())
            i = torch.tensor(X)
            Q[idx] = i
            idx += 1
        # Q_eigenvalues, Q_eigenvectors = np.linalg.eig(Q)  #   b x n
        # 对角线元素
        Q_eye = torch.eye(input_size, input_size)
        Q_diag = (Q * Q_eye).sum(1)  # b x n   x   n x n   = b x n  = b x n
        # index_anti = torch.linspace(input_size - 1, 0, input_size).unsqueeze(0).long()
        # Q_diag_anti = Q_diag.clone()  #
        # idx_q = 0
        # for i in Q:
        #     Q_diag_anti[idx_q] = i.gather(0, index_anti)
        #     idx_q += 1
        Q_row = Q.sum(2)  #
        Q_tran = torch.cat((p1, Q), dim=2)
        Q = torch.cat((p2, Q_tran), dim=1).to(device)  # b x n+1 x n+1
        # Q_eigenvalues = torch.cat((p0, torch.from_numpy(Q_eigenvalues)), dim=1).unsqueeze(1).to(
        #     device)  # b x 1 x n+1
        Q_diag = torch.cat((p0, Q_diag), dim=1).unsqueeze(1).to(device)  # b x n+1
        Q_row = torch.cat((p0, Q_row), dim=1).unsqueeze(1).to(device)  # b x 1 x n+1 #
        dynamic = dynamic.to(device)
        Q_RC = Q_row

        with torch.no_grad():
            time1 = time.time()
            tour_indices, _ = actor(Q, Q_diag, Q_RC, dynamic)
        time2 = time.time()
        time_ += time2 - time1
        reward = reward_fn(Q, tour_indices).mean().item()
        rewards.append(reward)
        if RLLS:
            time3 = time.time()

            DRLBISolution_reward= DRLBISolution(Q,tour_indices)
            time4=time.time()
            rewards_LS.append(DRLBISolution_reward)
            time_LS = time4-time3 +time_

    if RLLS:
        return np.mean(rewards), time_, np.mean(rewards_LS), time_LS
    return np.mean(rewards), time_


def train(actor, critic, task, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm, density, lb_linear, ub_linear, lb_quadr, ub_quadr,RLLS, train_seed,
          **kwargs):

    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    save_dir = os.path.join(task, '%d' % num_nodes, now)

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    actor_scheduler = lr_scheduler.MultiStepLR(actor_optim, range(5000, 5000 * 1000, 5000), gamma=float(0.96))

    train_loader = DataLoader(train_data, batch_size, True, num_workers=16)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=16)

    best_params = None
    best_reward = np.inf

    epoch_idx = 1

    for epoch in range(5):  # epoch 5
        torch.cuda.empty_cache()

        actor.train()
        critic.train()

        times, losses, rewards, critic_rewards = [], [], [], []

        epoch_start = time.time()
        start = epoch_start

        for batch_idx, batch in enumerate(train_loader):  # enumerate()
            x0, dynamic, Q = batch  # datasfile":('...','...','...')

            num_samples, input_size,_input_size = Q.size()
            p0 = torch.zeros(num_samples, 1)
            p1 = torch.zeros(num_samples, input_size, 1)
            p2 = torch.zeros(num_samples, 1, input_size + 1)

            idx = 0
            for i in Q:
                x = i.numpy()
                X = np.triu(x)
                X += X.T - np.diag(X.diagonal())
                i = torch.tensor(X)
                Q[idx] = i
                idx += 1
            # Q_eigenvalues, Q_eigenvectors = np.linalg.eig(Q)  #    b x n
            # 对角线元素
            Q_eye = torch.eye(input_size, input_size)
            Q_diag = (Q * Q_eye).sum(1)  # b x n   x   n x n   = b x n  = b x n
            index_anti = torch.linspace(input_size - 1, 0, input_size).unsqueeze(0).long()
            Q_diag_anti = Q_diag.clone()  #
            idx_q = 0
            for i in Q:
                Q_diag_anti[idx_q] = i.gather(0, index_anti)
                idx_q += 1
            Q_row = Q.sum(2)  #
            Q_tran = torch.cat((p1, Q), dim=2)
            Q = torch.cat((p2, Q_tran), dim=1).to(device)  # b x n+1 x n+1
            # Q_eigenvalues = torch.cat((p0, torch.from_numpy(Q_eigenvalues)), dim=1).unsqueeze(1).to(
            #     device)  # b x 1 x n+1
            Q_diag = torch.cat((p0, Q_diag), dim=1).unsqueeze(1).to(device)  # b x n+1
            Q_row = torch.cat((p0, Q_row), dim=1).unsqueeze(1).to(device)  # b x 1 x n+1 #
            dynamic = dynamic.to(device)
            Q_RC = Q_row

            tour_indices, tour_logp = actor(Q, Q_diag, Q_RC, dynamic)
            reward = reward_fn(Q, tour_indices)

            critic_est = critic(Q_diag, Q_RC).view(-1)

            advantage = -(reward - critic_est)

            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            actor_scheduler.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            critic_rewards.append(torch.mean(critic_est.detach()).item())
            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())

            if (batch_idx + 1) % 100 == 0:
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-100:])
                mean_reward = np.mean(rewards[-100:])

                print('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
                      (batch_idx, len(train_loader), mean_reward, mean_loss,
                       times[-1]))

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic.state_dict(), save_path)

        valid_dir = os.path.join(save_dir, '%s' % epoch)
        if RLLS:
            mean_valid, t_,mean_RL_LS,t1= validate(valid_loader, actor,RLLS, reward_fn, render_fn,
                                      valid_dir, num_plot=5)
            print("RL_LS:", mean_RL_LS)
        else:
            mean_valid, t_= validate(valid_loader, actor,RLLS, reward_fn, render_fn,
                                  valid_dir, num_plot=5)
        if mean_valid < best_reward:
            best_reward = mean_valid

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

        print('%2.4f Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, took: %2.4fs ' \
              '(%2.4fs / 100 batches)\n' % \
              (epoch_idx, mean_loss, mean_reward, mean_valid, time.time() - epoch_start,
               np.mean(times)))
        epoch_idx += 1


# 对模型的训练
def train_ubqp(args):
    from tasks import ubqp
    from tasks.ubqp import WTdataset

    STATIC_SIZE = 2
    DYNAMIC_SIZE = 2
    test_seed = args.test_seed
    time1 = time.time()
    train_data = WTdataset(args.train_size,
                           args.num_nodes,
                           args.seed)
    valid_data = WTdataset(args.valid_size,
                           args.num_nodes,
                           args.seed + 2)
    time2 = time.time()
    print(time2 - time1)
    actor = DRL4UBQP(args.num_nodes + 1, STATIC_SIZE, args.hidden_size, DYNAMIC_SIZE, args.num_layers, args.n_head,
                     args.n_layers, args.k_dim,
                     args.v_dim, args.const_local, args.pos, train_data.update_dynamic, train_data.update_mask,
                     args.dropout
                     ).to(device)

    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)
    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = ubqp.reward
    kwargs['render_fn'] = None  # sat.render

    if not args.test:
        try:
            train(actor, critic, **kwargs)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise e

    if not args.instance:
        test_data = WTdataset(args.valid_size,
                              args.num_nodes,
                              args.seed + 2)
        test_dir = 'test'
        test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
        if args.checkpoint:
            path = os.path.join(args.checkpoint, 'actor.pt')
            actor.load_state_dict(torch.load(path, device))
            path = os.path.join(args.checkpoint, 'critic.pt')
            critic.load_state_dict(torch.load(path, device))
            out, time1, out_ls, time_ls = validate(test_loader, actor, True, ubqp.reward, None, test_dir, test_seed, num_plot=5)
            print('RL Average rewards: ', out)
            print("RL_LS:", out_ls)
            print('RL  times: ', time1)
            print("RL_LS  times:", time_ls)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--train_seed', default=1, type=int)
    parser.add_argument('--test_seed', default=2, type=int)
    # data distribution accroding to ...
    parser.add_argument('--pos', default=0.2, type=float)
    parser.add_argument('--density', default=1, type=float)
    parser.add_argument('--lb_linear', default=1, type=float)
    parser.add_argument('--ub_linear', default=1, type=float)
    parser.add_argument('--lb_quadr', default=1, type=float)
    parser.add_argument('--ub_quadr', default=1, type=float)
    # Net parameters
    parser.add_argument('--RLLS', default=False)

    parser.add_argument('--instance', default=False)
    # parser.add_argument('--checkpoint', default=None)
    # parser.add_argument('--test', action='store_true', default=False)
    # parser.add_argument('--instance', default=True)
    parser.add_argument('--checkpoint', default="ubqp/150/1")
    parser.add_argument('--test', action='store_true', default=True)
    parser.add_argument('--n_head', default=8, type=int)
    parser.add_argument('--n_layers', default=3, type=int)
    parser.add_argument('--k_dim', default=256, type=int)
    parser.add_argument('--v_dim', default=256, type=int)
    parser.add_argument('--const_local', default=32, type=int)
    parser.add_argument('--task', default='ubqp')

    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--nodes', dest='num_nodes', default=50, type=int)
    parser.add_argument('--train-size', default=1,type=int)
    parser.add_argument('--valid-size', default=1000, type=int)
    args = parser.parse_args()
    if args.task == 'ubqp':
        train_ubqp(args)
    else:
        raise ValueError('Task <%s> not understood' % args.task)
