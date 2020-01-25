import numpy as np
import random
import matplotlib.pyplot as plt
import argparse



def sigmoid(z):
	return 1.0/(1+np.exp(-z))


def sigmoid_der(z):
	return z*(1.0-z)



class network:
	def __init__(self, input,des_output,nodes):
		self.input=input
		self.des_output=des_output	
		self.output=np.zeros(self.des_output.shape)
		self.nodes=nodes

	
	def make_weights(self):
		self.weights=[]
		if self.nodes[0]!=input.shape[1]:
			raise ValueError("The length ({0}) of the input values and the number ({1}) of input nodes should be the same!!".format(input.shape[1], self.nodes[0] ))

		for i in range(0, len(self.nodes)-1):
			self.weights.append(np.random.rand(self.nodes[i], self.nodes[i+1]))
#		print(self.weights)
		


	def feedforward(self):
		l=0
		self.layers=[]
		self.layers.append(self.input)
		for i in self.weights:
			self.layers.append(sigmoid(np.dot(self.layers[l],i)))			
			l+=1

		
		self.output=self.layers[-1]


	def feedforward2(self,weights):
		l=0
		self.layers=[]
		self.layers.append(self.input)
		for i in weights:
			self.layers.append(sigmoid(np.dot(self.layers[l],i)))
			l+=1


		self.output=self.layers[-1]
		



	def backpropagation(self):
		
		keep_update={}
		cost_function=2*(self.des_output-self.output )
		updates=len(self.layers)-1 ## we remove the input, because we kept it
		w=-1##since weight1 will never be used in the backprogation, it will always be the last step
		l=-2##because l[-1] is the output
		for i in range(0, len(self.layers)-1):
			
	
			if w==-1:
				

				new_weights_last=np.dot(self.layers[l].T, sigmoid_der( self.output) * cost_function)
#				print new_weights_last
				self.weights[w]+=new_weights_last				
				keep_update[updates]=sigmoid_der( self.output) * cost_function

	
									
				w-=1
				l-=1
				updates-=1
				
					
			
			else:

				new_weights=np.dot(self.layers[l].T, sigmoid_der( self.layers[l+1])* np.dot(keep_update[updates+1], self.weights[w+1].T))
#				print new_weights
				self.weights[w]+=new_weights
				keep_update[updates]= sigmoid_der(self.layers[l+1])*np.dot(keep_update[updates+1], self.weights[w+1].T)
                		 
				l-=1
				w-=1
				updates-=1


#		print self.weights

	def get_weights(self):
		return self.weights
	
	def get_output(self):
		return self.output

	def get_loss(self):
		 return  1/float(self.input.shape[0])*sum(np.subtract(self.des_output,self.output)**2)	
		



parser=argparse.ArgumentParser(description= """
            Description
            -----------
           Generate neural network

            Authors
            -----------
            Vlachos Christos""",formatter_class=argparse.RawDescriptionHelpFormatter)




parser.add_argument("--nodes", nargs='+', type=int, required=True,dest="nodes",default=None, help="List of nodes, .e.g 5 4 3 1 means 5 input nodes, two hidden layers with 4 and 3 nodes and one output node")
parser.add_argument("--training-sets", type=int,required=False, default=1,dest="train", help="Number of random training sets")
parser.add_argument("--iterations", type=int, required=True, dest="iter",help="Number of iterations through the network")

args = parser.parse_args()


#nodes=[5,4,3,1]
input=np.random.rand(args.train,args.nodes[0])
print("My input nodes({0} nodes for {1} training set(s) (each row is a training set))  are:\n".format(args.nodes[0], args.train), input)


des_output=np.random.rand(args.train,args.nodes[-1])
des_output=np.around(des_output)
print("The desired output for each training set is (either 0 or 1)\n", des_output)
print("\n")




mynetwork=network(input,des_output,args.nodes)
mynetwork.make_weights()
#print mynetwork.make_weights()
losses=[]



for i in range(0,args.iter):
	
        mynetwork.feedforward()
        mynetwork.backpropagation()


        if i==0 or i==50 or i==100 or i==300 or i==500 or i==1000 or i ==5000 or i==10000 or i==50000:
                print("After {0} iterations my networks returns:\n".format(i), mynetwork.get_output())
                print("\n")
        loss=mynetwork.get_loss()
        losses.append(loss[0])


        if i==args.iter-1:
        	weights=mynetwork.get_weights()


print("After {0} iterations my network returns:\n".format(args.iter), mynetwork.output)
#plt.plot(range(0,args.iter,100), losses[0::100])
#plt.title("Cost function (difference between expected and estimated output)")
#plt.show()

print("#########")
print("Make a new network with the same random input and feed it with the parameters estimated from the training set")

nn=network(input, des_output, args.nodes)
print("The output of the test network is:")
nn.feedforward2(weights)

print(nn.output)
print("If the output is the same as before then we have a trained network for this input!!")
