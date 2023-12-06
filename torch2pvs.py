import torch
import torch.nn as nn
import numpy
import argparse

# Constant blocks



const_header = ''': THEORY
    BEGIN 
        IMPORTING matrices@matrices
        IMPORTING lnexp@exp_series
'''

const_trailer = "END "

const_mat_relu = '''
    act(M: Matrix): MatrixMN(rows(M),columns(M)) =
        form_matrix(LAMBDA (i,j:nat): actFun(entry(M)(i,j)), rows(M), columns(M));
'''


const_network = "net(input: Matrix): Matrix ="


sym_vars = {}

def getLeakyReluString(slope):
    return f'lrelu(x: real): real = IF x > 0 THEN x ELSE {slope}*x ENDIF'

def genAxiomsFromSymVars():
    axioms = []
    for k in sym_vars.keys():
        axioms.append(k+"_ax: AXIOM "+k+"="+str(sym_vars[k]))
    return axioms

def toPVSMatrix(tensor):
    pvs_matrix_entries = []
    for row in tensor:
        np_weight = row.detach().numpy()
        np_str_weight = numpy.char.mod("%f", np_weight)
        joined_str_weight = ','.join(np_str_weight)
        pvs_formatted = "(:"+joined_str_weight+":)"
        pvs_matrix_entries.append(pvs_formatted)
    pvs_formatted_matrix = "(:"+','.join(pvs_matrix_entries)+":)"
    return pvs_formatted_matrix

def toPVSSymMatrix(tensor, prefix):
    pvs_matrix_entries = []
    for r,row in enumerate(tensor):
        np_weight = row.detach().numpy()
        weight_num = len(np_weight)
        weight_syms = []
        for i,w in enumerate(np_weight):
            weight_syms.append(prefix+str(r)+str(i))
            sym_vars[(prefix+str(r)+str(i))] = w
        joined_str_weight = ','.join(weight_syms)
        pvs_formatted = "(:"+joined_str_weight+":)"
        pvs_matrix_entries.append(pvs_formatted)
        
    pvs_formatted_matrix = "(:"+','.join(pvs_matrix_entries)+":)"
    
    return pvs_formatted_matrix    

def genMatrixDeclarations(model, sym=False):
    matrix_declarations = []
    for i,layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            weights = layer.weight
            bias  = layer.bias

            # Increase bias dimensions by 1 if 1d
            if len(bias.shape) == 1:
                bias = torch.Tensor(numpy.array([bias.detach().numpy()]))

            # Weights must be transposed to match matrix matrix sizes
            tr_weights = torch.transpose(weights, 0, 1)

            w_cols = len(weights)
            w_rows = len(weights[0])

            b_cols = len(bias)
            b_rows = len(bias[0])


            pvs_formatted_weight = toPVSSymMatrix(tr_weights,"l"+str(i)+"_w")
            pvs_fullweight_entry = "linear"+str(i)+": MatrixMN("+str(w_rows)+","+str(w_cols)+") = "+pvs_formatted_weight

            pvs_formatted_bias = toPVSSymMatrix(bias,"l"+str(i)+"_b")
            pvs_fullbias_entry = "linear_bias"+str(i)+": MatrixMN("+str(b_cols)+","+str(b_rows)+") = "+pvs_formatted_bias


            matrix_declarations.append(pvs_fullweight_entry)
            matrix_declarations.append(pvs_fullbias_entry)
    return matrix_declarations

def genNetworkOperationSequence(model):
    network_operations = []
    activation_idx_start = 0
    num_layers = len(model)

    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            if i == 0:
                network_operations.append("input")
            network_operations.append("*linear"+str(i)+"+linear_bias"+str(i))
        
        if isinstance(layer, nn.ReLU):

            # Insert relu at beginning of sequence and close bracket at end of sequence
            network_operations.insert(0,"reluMat(")
            network_operations.append(")")
        
        if isinstance(layer, nn.LeakyReLU):
            network_operations.insert(0,"lreluMat(")
            network_operations.append(")")

        if isinstance(layer, nn.Tanh):
            network_operations.insert(0,"tanhMat(")
            network_operations.append(")")

    return network_operations

def genTheorem(name, input_vars, output_vars):
    name+=": THEOREM\n"
    name+="\t\tFORALL ("

    xinvars = []
    xnames = []
    for i in range(input_vars):
        xnames.append("x"+str(i)+"in")
        xinvars.append("x"+str(i)+"in: x"+str(i)+"inreal")

    name+=','.join(xinvars)+"):\n"
    name+="\t\t\tentry( net( (:(:"+','.join(xnames)+":):) ) )(0,0) <TBD>"

    return name

def genConstraintExpressions(input_vars):
    constraints = ""
    for i in range(input_vars):
        constraints+="\tx"+str(i)+"inreal: TYPE = { r: real | r>=-<TBD> AND r<=<TBD> }\n"

    return constraints

# Read input arguments

parser = argparse.ArgumentParser(
                    prog='torch2pvs',
                    description='Generate a skeleton PVS theory for a MLP torch model',
                    epilog='Federico Rossi, 2023')

parser.add_argument("model_path")
parser.add_argument("-o","--output_vars", default=1, required=False, type=int)
parser.add_argument("-p","--pvs_output", default=None, required=False)
parser.add_argument("-n","--name", default="mlp", required=False)
args = parser.parse_args()

model = torch.load(args.model_path)

input_vars = next(model.parameters()).size()[1]

lines = genMatrixDeclarations(model)
sequence = genNetworkOperationSequence(model)

theorem = genTheorem("network_bounds",input_vars,args.output_vars)

constraints = genConstraintExpressions(input_vars)

wb_axioms = genAxiomsFromSymVars()

# Emit PVS

pvs_buffer = ""
pvs_buffer+= "%"+str(model).replace("\n","\n%")+"\n"
pvs_buffer+=args.name+const_header


pvs_buffer+="\n\t" + ",".join(sym_vars.keys())+": VAR real"
pvs_buffer+="\n\t" + "\n\t".join(wb_axioms)

pvs_buffer+="\n\t"
for line in lines:
    pvs_buffer+=line+"\n\t"


pvs_buffer+="\n"+const_mat_relu


pvs_buffer+="\n"+constraints

pvs_buffer+="\n\t"+const_network
pvs_buffer+="\t\t\n\t\t"+"".join(sequence)+"\n\n"

pvs_buffer+="\t"+theorem
pvs_buffer+="\n\n"+const_trailer+args.name

if args.pvs_output is not None:
    outfile = open(args.pvs_output, "w")
    outfile.write(pvs_buffer)
else:
    print(pvs_buffer)
