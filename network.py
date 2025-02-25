import numpy as np
import numpy
import copy
import pickle
import matplotlib.pyplot as plt
from rich.progress import Progress
from rich import print as rprint
import struct

class Network:
    def __init__(self,neuronCnt:list,trainSet:list,validSet:list,activation:list)->None:
        self.n=neuronCnt
        self.trainSet=trainSet
        self.validSet=validSet
        self.activation=activation
        self.para=[]
        self.a=[]
        self.z=[]
        self.onehot=np.eye(self.n[-1])
        self.allTrainData=np.array([trainSet[i]["data"] for i in range(len(trainSet))])
        self.allValidData=np.array([validSet[i]["data"] for i in range(len(validSet))])
        self.allTrainLabel=np.array([trainSet[i]["label"] for i in range(len(trainSet))])
        self.allValidLabel=np.array([validSet[i]["label"] for i in range(len(validSet))])
        self.allTrainOnehot=np.array([self.onehot[trainSet[i]["label"]] for i in range(len(trainSet))])
        self.allValidOnehot=np.array([self.onehot[validSet[i]["label"]] for i in range(len(validSet))])
    
    def setPara(self,seed:int=42,para:list=None)->None:
        np.random.seed(seed)
        if para is None:
            distribution=[
                    {"b":(0,0)}
                if i==0 else 
                    {"b":(0,0),"w":(-np.sqrt(6/(self.n[i-1]+self.n[i])),np.sqrt(6/(self.n[i-1]+self.n[i])))} 
                for i in range(len(self.n))
            ]
            self.para=[{
                key:np.random.rand(self.n[l])*(value[1]-value[0])+value[0]
                if key=="b" else 
                np.random.rand(self.n[l],self.n[l-1])*(value[1]-value[0])+value[0]
                for key,value in distribution[l].items()
            } for l in range(len(self.n))]
        else:
            self.para=copy.deepcopy(para)
        self.a=[np.zeros(self.n[i]) for i in range(len(self.n))]
        self.z=[np.zeros(self.n[i]) for i in range(len(self.n))]
        # print(self.para)
    
    def forwardPass(self,data:np.ndarray,para:list=None)->None:
        if para is None:
            para=self.para
        for i in range(len(self.n)):
            if i==0:
                self.z[0]=data.reshape(-1)+para[i]["b"]
            else:
                self.z[i]=para[i]["w"]@self.a[i-1]+para[i]["b"]
            self.a[i]=self.activation[i](self.z[i])
    
    def multiForwardPass(self,data:np.ndarray,para:list=None)->None:
        if para is None:
            para=self.para
        self.a=[np.zeros((data.shape[0],self.n[i])) for i in range(len(self.n))]
        self.z=[np.zeros((data.shape[0],self.n[i])) for i in range(len(self.n))]
        for i in range(len(self.n)):
            if i==0:
                self.z[0]=data.reshape(data.shape[0],self.n[0])+para[i]["b"]
            else:
                self.z[i]=np.einsum("ij,kj->ki",para[i]["w"],self.a[i-1])+para[i]["b"]
            self.a[i]=np.apply_along_axis(self.activation[i],1,self.z[i])
    
    def possibility(self,data:np.ndarray,para:list=None)->np.ndarray:
        if para is None:
            para=self.para
        self.forwardPass(data,para)
        return self.a[-1]
    
    def predict(self,data:np.ndarray,para:list=None):
        if para is None:
            para=self.para
        return self.possibility(data,para).argmax()
    
    def trainLossAccu(self,para:list=None)->float:
        if para is None:
            para=self.para
        self.multiForwardPass(self.allTrainData,para)
        loss=np.sum(((self.a[-1]-self.allTrainOnehot)**2)/2,axis=1)
        acuu=np.argmax(self.a[-1],axis=1)
        return np.mean(loss),np.mean((acuu==self.allTrainLabel).astype(float))
    
    def validLossAccu(self,para:list=None)->float:
        if para is None:
            para=self.para
        self.multiForwardPass(self.allValidData,para)
        loss=np.sum(((self.a[-1]-self.allValidOnehot)**2)/2,axis=1)
        acuu=np.argmax(self.a[-1],axis=1)
        return np.mean(loss),np.mean((acuu==self.allValidLabel).astype(float))
    
    def getGrad(self,trainRange:list,para:list=None)->list:
        if para is None:
            para=self.para
        grad=[
                {"b":np.zeros(self.n[0])}
            if i==0 else 
                {"b":np.zeros(self.n[i]),"w":np.zeros((self.n[i],self.n[i-1]))}
            for i in range(len(self.n))
        ]
        delta=[np.zeros((len(trainRange),self.n[i])) for i in range(len(self.n))]
        multiData=self.allTrainData[trainRange]
        multiOnehot=self.allTrainOnehot[trainRange]
        self.multiForwardPass(multiData,para)
        if self.activation[-1].type=="dot":
            delta[-1]=np.einsum("kij,kj->ki",
                np.apply_along_axis(self.activation[-1].d,1,self.z[-1]),
                self.a[-1]-multiOnehot)
        else:
            delta[-1]=np.einsum("ki,ki->ki",
                np.apply_along_axis(self.activation[-1].d,1,self.z[-1]),
                self.a[-1]-multiOnehot)
        for i in range(len(self.n)-2,-1,-1):
            if self.activation[i].type=="dot":
                delta[i]=np.einsum("lij,kj,lk->li",
                    np.apply_along_axis(self.activation[i].d,1,self.z[i]),
                    para[i+1]["w"],delta[i+1])
            else:
                delta[i]=np.einsum("li,ji,lj->li",
                    np.apply_along_axis(self.activation[i].d,1,self.z[i]),
                    para[i+1]["w"],
                    delta[i+1])
        for i in range(len(self.n)):
            for key in grad[i]:
                if key=="b":
                    grad[i][key]=np.mean(delta[i],axis=0)
                if key=="w":
                    grad[i][key]=np.mean(np.einsum("li,lk->lik",delta[i],self.a[i-1]),axis=0)
        return grad
            
    def getGrad_old(self,trainRange:list,para:list=None)->list:
        if para is None:
            para=self.para
        grad=[
                {"b":np.zeros(self.n[0])}
            if i==0 else 
                {"b":np.zeros(self.n[i]),"w":np.zeros((self.n[i],self.n[i-1]))}
            for i in range(len(self.n))
        ]
        for id in trainRange:
            singleSet=trainSet[id]
            self.forwardPass(singleSet["data"],para)
            delta=[np.zeros(self.n[i]) for i in range(len(self.n))]
            if self.activation[-1].type=="dot":
                delta[-1]=np.einsum("ij,j->i",self.activation[-1].d(self.z[-1]),self.a[-1]-self.onehot[singleSet["label"]])
            else:
                delta[-1]=np.einsum("i,i->i",self.activation[-1].d(self.z[-1]),self.a[-1]-self.onehot[singleSet["label"]])
            for i in range(len(self.n)-2,-1,-1):
                if self.activation[i].type=="dot":
                    delta[i]=np.einsum("ij,kj,k->i",self.activation[i].d(self.z[i]),para[i+1]["w"],delta[i+1])
                else:
                    delta[i]=np.einsum("i,ji,j->i",self.activation[i].d(self.z[i]),para[i+1]["w"],delta[i+1])
            for i in range(len(self.n)):
                for key in grad[i]:
                    if key=="b":
                        grad[i][key]+=delta[i]
                    if key=="w":
                        grad[i][key]+=np.einsum("i,k->ik",delta[i],self.a[i-1])
        for i in range(len(self.n)):
            for key in grad[i]:
                grad[i][key]/=len(trainRange)
        return grad
    
    def updatePara(self,paraGrad:list,learnRate:float)->None:
        for i in range(len(self.para)):
            for key in self.para[i]:
                self.para[i][key]-=paraGrad[i][key]*learnRate
    
    def trainBatch(self,trainRange:list,learnRate:float)->None:
        grad=self.getGrad(trainRange)
        self.updatePara(grad,learnRate)
        
    
    def trainEpoch(self,batchSize:int,learnRate:float,reportFrec:int)->None:
        with Progress() as progress:
            task = progress.add_task("[cyan]Training...", total=len(trainSet))
            i=0
            for s in range(0,len(trainSet),batchSize):
                progress.update(task, advance=batchSize)
                self.trainBatch(range(s,min(len(trainSet),s+batchSize)),learnRate)
                i+=1
                if reportFrec>0 and i%reportFrec==0:
                    print()
                    print(f"{min(len(trainSet),s+batchSize)}/{len(trainSet)}")
                    print(f"    train:{nn.trainLossAccu()}\n    valid:{nn.validLossAccu()}")
                
                with open("para.pkl","wb") as f:
                    pickle.dump(nn.para,f)
    
if __name__ == "__main__":

    def x(x:np.ndarray)->np.ndarray:
        return 2*x
    def xD(x:np.ndarray)->np.ndarray:
        return np.array([2 for i in range(len(x))])

    x.d=xD
    x.type="times"

    def sigmoid(x:np.ndarray)->np.ndarray:
        return 1/(1+np.exp(-x))
    def sigmoidD(x:np.ndarray)->np.ndarray:
        sm=sigmoid(x)
        return sm*(1-sm)

    sigmoid.d=sigmoidD
    sigmoid.type="times"

    def tanh(x:np.ndarray)->np.ndarray:
        return np.tanh(x)
    def tanhD(x:np.ndarray)->np.ndarray:
        return 1-np.tanh(x)**2

    tanh.d=tanhD
    tanh.type="times"

    def softmax(x:np.ndarray)->np.ndarray:
        xExp=np.exp(x-np.max(x))
        return xExp/np.sum(xExp)
    def softmaxD(x:np.ndarray)->np.ndarray:
        sf=softmax(x)
        return np.diag(sf)-np.outer(sf,sf)

    softmax.d=softmaxD
    softmax.type="dot"
    
    # data=np.load(r'./mnist.npz')
    
    # trainValidData=data["x_train"]
    # trainValidLabel=data["y_train"]
    # testData=data["x_test"]
    # testLabel=data["y_test"]
    
    with open(r"./train-images-idx3-ubyte","rb") as f:
        struct.unpack(">4i",f.read(16))
        trainValidData=np.fromfile(f,dtype=np.uint8).reshape(-1,784)
    with open(r"./t10k-images-idx3-ubyte","rb") as f:
        struct.unpack(">4i",f.read(16))
        testData=np.fromfile(f,dtype=np.uint8).reshape(-1,784)
    with open(r"./train-labels-idx1-ubyte","rb") as f:
        struct.unpack(">2i",f.read(8))
        trainValidLabel=np.fromfile(f,dtype=np.uint8)
    with open(r"./t10k-labels-idx1-ubyte","rb") as f:
        struct.unpack(">2i",f.read(8))
        testLabel=np.fromfile(f,dtype=np.uint8)
    
    # for i in range(len(trainValidData)):
    #     plt.imshow(trainValidData[i].reshape(28,28),cmap='gray')
    #     print(trainValidLabel[i])
    #     plt.show()
        
    
    
    np.random.seed(42)
    trainSet=[{"data":np.array(trainValidData[i])/255,"label":trainValidLabel[i]} for i in range(50000)]
    validSet=[{"data":np.array(trainValidData[i])/255,"label":trainValidLabel[i]} for i in range(50000,60000)]
    testSet=[{"data":np.array(testData[i])/255,"label":testLabel[i]} for i in range(10000)]
    
    n=[784,32,32,32,32,10]
    
    activation=[tanh,tanh,tanh,tanh,tanh,softmax]
    
    nn=Network(n,trainSet,validSet,activation)
    
    
    k=input("use trained para?y/n\n")
    
    if k=="y" or k=="Y":
        with open("para.pkl","rb") as f:
            para=pickle.load(f)
        nn.setPara(para=para)
    else:
        nn.setPara()
        with open("para.pkl","wb") as f:
            pickle.dump(nn.para,f)
    
    # print(nn.para)
    
    if k!="Y":
        print(f"train:{nn.trainLossAccu()}\nvalid:{nn.validLossAccu()}")
    
    while True:
        epochCnt=int(input("Epoch:"))
        if(epochCnt<=0):
            break
        batchSize=int(input("BatchSize:"))
        reportFrec=int(input("ReportFrec:"))
        learnRate=float(input("LearnRate:"))
        
        for epoch in range(epochCnt):
            print(f"Epoch {epoch+1}/{epochCnt} training!")
            nn.trainEpoch(batchSize,learnRate,reportFrec)
            print(f"Epoch {epoch+1}/{epochCnt} trained!")
            # print(f"train:{nn.trainLossAccu()}\nvalid:{nn.validLossAccu()}")
            
            with open("para.pkl","wb") as f:
                pickle.dump(nn.para,f)
        
        print(f"train:{nn.trainLossAccu()}\nvalid:{nn.validLossAccu()}")
        print()
        print()
    
    fashion={
        0:"T恤",
        1:"裤子",
        2:"套衫",
        3:"裙子",
        4:"外套",
        5:"凉鞋",
        6:"汗衫",
        7:"运动鞋",
        8:"包",
        9:"踝靴",
    }
    
    while True:
        id=int(input("id:"))
        if id==-1:
            break
        if id==-2:
            id=np.random.randint(0,len(testSet))
            print(id)
        if id==-3:
            id=np.random.randint(0,len(testSet))
            for i in range(len(testSet)):
                poss=nn.possibility(testSet[(id+i)%len(testSet)]["data"])
                if poss.argmax()!=testSet[(id+i)%len(testSet)]["label"]:
                    id=(id+i)%len(testSet)
                    break
            print(id)
        id%=len(testSet)
        poss=nn.possibility(testSet[id]["data"])
        for i in range(len(poss)):
            rprint(f'{fashion[i]}\t:{poss[i]:.6f} {"[green]█[/green]"*int(poss[i]*100)}{" <-label" if testSet[id]["label"]==i else ""}{" <-predict" if poss.argmax()==i else ""}')
        print(f"predict\t:{fashion[poss.argmax()]}")
        print(f'label\t:{fashion[testSet[id]["label"]]}')
        plt.imshow(testSet[id]["data"].reshape(28,28),cmap='gray')
        plt.show()
        print()
    
    
    # 小测试
    
    # trainSet=[{"data":np.array([1,1,1]),"label":0},{"data":np.array([0,0,0]),"label":1}]
    
    # para=[{"b":np.array([1,2,3])},{"b":np.array([2,3]),"w":np.array([[0,1,2],[3,4,5]])}]
    # nn=Network([3,2],trainSet,trainSet,[x,x])
    # nn.setPara(42,para)
    
    # data=np.array([[1,1,1],[0,0,0]])
    # nn.forwardPass(data[0],para)
    # print("para:",para)
    # print("z:",nn.z)
    # print("a:",nn.a)
    # nn.multiForwardPass(data,para)
    # print("para:",para)
    # print("z:",nn.z)
    # print("a:",nn.a)
    
    
    # print(nn.getGrad_old(range(1),para))
    # print(nn.getGrad_old(range(1,2),para))
    # print(nn.getGrad(range(2),para))