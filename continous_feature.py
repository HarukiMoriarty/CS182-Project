import numpy as np
import torch

with open("res.txt","r") as f:
    annos = f.readlines()
annos = [np.array([float(x) for x in anno.split()]) for anno in annos]
annos = np.stack(annos)

pattern = "generate_with_latent/seed{}.npy"
xs = []
for idx in range(1,len(annos)+1):
    x = np.load(pattern.format(str(idx).zfill(4)))
    xs.append(x[:,4:8])
xs = np.concatenate(xs).reshape(len(annos),-1)

types_num=11
samples_num=xs.shape[0]
features_num=xs.shape[1]

X=[[] for i in range(types_num)]
for i in range(samples_num):
    X[int(annos[i][0]*10)].append(xs[i])

m=np.zeros((types_num,features_num))
for i in range(types_num):
    for x in X[i]:
        m[i]+=x
    if len(X[i])!=0:
        m[i]/=len(X[i])
    else:
        print("There is no point in some type.")
    print(len(X[i]))
m_avg=m.mean(axis=0)

S=np.zeros((features_num,features_num))
V=np.zeros((features_num,features_num))
for i in range(types_num):
    for j in range(len(X[i])):
        now=X[i][j]-m[i]
        S+=now.reshape(now.shape[0],1)@now.reshape(1,now.shape[0])
print(S)

for i in range(types_num):
    now=m[i]-m_avg
    V+=now.reshape(now.shape[0],1)@now.reshape(1,now.shape[0])
print(V)

try:
    print(np.linalg.matrix_rank(S))
    Sinv=np.linalg.inv(S)
except:
    print('qwq')
    Sinv=np.linalg.inv(S+0.0001*np.eye(S.shape[0]))
lam,mu=np.linalg.eig(Sinv@V)

ans=0
maxl=-1e9
for x in lam:
    if abs(x.imag)<=0.0001 and x.real>maxl:
        ans=mu[i]
        maxl=x
print(lam,maxl,'\n',ans)

sleeve_vector = np.real(ans[:512]).reshape(1,512)
sleeve_vector /= np.linalg.norm(sleeve_vector,ord=2)
torch.save(torch.tensor(sleeve_vector), "sleeve_length.pt")