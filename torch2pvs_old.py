import torch
import torch.nn as nn
import numpy
import argparse

# Constant blocks

# Header for theory imports,
# add new imports here
const_header = ''': THEORY
    BEGIN 
        IMPORTING matrices@matrices
        IMPORTING lnexp@exp_series
'''

const_trailer = "END "

# Activation function placeholder
const_mat_act = '''
    % Replace actFun with a scalar function real -> real
    act(M: Matrix): MatrixMN(rows(M),columns(M)) =
        form_matrix(LAMBDA (i,j:nat): actFun(entry(M)(i,j)), rows(M), columns(M));
'''

const_network = "net(input: Matrix): Matrix ="

sym_vars = {}


# Generates the PVS function signature for a leaky relu with slope 'slope'
def get_leaky_relu_string(slope):
    return f'lrelu(x: real): real = IF x > 0 THEN x ELSE {slope}*x ENDIF'


# Generates the list of axioms used to replace symbolic weights
# only when script called with '--sym' flag
def gen_axioms_from_sym_vars():
    axioms = []
    for k in sym_vars.keys():
        axioms.append(k + "_ax: AXIOM " + k + "=" + str(sym_vars[k]))
    return axioms


# Generates a PVS matrix signature from the pytorch tensor 'tensor'
def to_pvs_matrix(tensor):
    pvs_matrix_entries = []
    for row in tensor:
        np_weight = row.detach().numpy()
        np_str_weight = numpy.char.mod("%f", np_weight)
        joined_str_weight = ','.join(np_str_weight)
        pvs_formatted = "(:" + joined_str_weight + ":)"
        pvs_matrix_entries.append(pvs_formatted)
    pvs_formatted_matrix = "(:" + ','.join(pvs_matrix_entries) + ":)"
    return pvs_formatted_matrix


# Generates a PVS matrix from the tensor 'tensor'
# using only symbolic weights, with each symbol prefixed by 'prefix'
def to_pvs_sym_matrix(tensor, prefix):
    pvs_matrix_entries = []
    for r, row in enumerate(tensor):
        np_weight = row.detach().numpy()

        weight_syms = []
        for i, w in enumerate(np_weight):
            weight_syms.append(prefix + str(r) + str(i))
            sym_vars[(prefix + str(r) + str(i))] = w
        joined_str_weight = ','.join(weight_syms)
        pvs_formatted = "(:" + joined_str_weight + ":)"
        pvs_matrix_entries.append(pvs_formatted)

    pvs_formatted_matrix = "(:" + ','.join(pvs_matrix_entries) + ":)"

    return pvs_formatted_matrix


# Generates all matrices PVS declaration from the model 'model'
# If 'sym' is True then the matrices declaration contains symbolic weights
def gen_matrix_declarations(model_, sym=False):
    matrix_declarations = []
    for i, layer in enumerate(model_):
        if isinstance(layer, nn.Linear):
            weights = layer.weight
            bias = layer.bias
            print(bias)
            # Increase bias dimensions by 1 if 1d
            if bias is not None and len(bias.shape) == 1:
                bias = torch.Tensor(numpy.array([bias.detach().numpy()]))

            # Weights must be transposed to match matrix  sizes
            tr_weights = torch.transpose(weights, 0, 1)

            w_cols = len(weights)
            w_rows = len(weights[0])

            if bias is not None:
                b_cols = len(bias)
                b_rows = len(bias[0])

            if sym:
                pvs_formatted_weight = to_pvs_sym_matrix(tr_weights, "l" + str(i) + "_w")
            else:
                pvs_formatted_weight = to_pvs_matrix(tr_weights)

            pvs_full_weight_entry = "linear" + str(i) + ": MatrixMN(" + str(w_rows) + "," + str(
                w_cols) + ") = " + pvs_formatted_weight

            if bias is not None:
                if sym:
                    pvs_formatted_bias = to_pvs_sym_matrix(bias, "l" + str(i) + "_b")
                else:
                    pvs_formatted_bias = to_pvs_matrix(bias)

                pvs_full_bias_entry = "linear_bias" + str(i) + ": MatrixMN(" + str(b_cols) + "," + str(
                    b_rows) + ") = " + pvs_formatted_bias
                matrix_declarations.append(pvs_full_bias_entry)

            matrix_declarations.append(pvs_full_weight_entry)
    return matrix_declarations


# Generates the succession of function calls to perform network inference
# for the model 'model'
def gen_network_operation_sequence(model_):
    network_operations = []

    for i, layer in enumerate(model_):
        if isinstance(layer, nn.Linear):
            if i == 0:
                network_operations.append("input")
            network_operations.append("*linear" + str(i) + "+linear_bias" + str(i))

        if isinstance(layer, nn.ReLU):
            # Insert relu at beginning of sequence and close bracket at end of sequence
            network_operations.insert(0, "reluMat(")
            network_operations.append(")")

        if isinstance(layer, nn.LeakyReLU):
            network_operations.insert(0, "lreluMat(")
            network_operations.append(")")

        if isinstance(layer, nn.Tanh):
            network_operations.insert(0, "tanhMat(")
            network_operations.append(")")

    return network_operations


# Generates a placeholder theorem
def gen_theorem(name_, input_vars_):
    name_ += ": THEOREM\n"
    name_ += "\t\tFORALL ("

    x_input_vars = []
    x_input_names = []
    for i in range(input_vars_):
        x_input_names.append("x" + str(i) + "in")
        x_input_vars.append("x" + str(i) + "in: x" + str(i) + "inreal")

    name_ += ','.join(x_input_vars) + "):\n"
    name_ += "\t\t\tentry( net( (:(:" + ','.join(x_input_names) + ":):) ) )(0,0) <TBD>"

    return name_


# Generates placeholder constraints on input variables 'input_vars'
def gen_constraint_expressions(input_vars_l):
    constraints_ = ""
    for i in range(input_vars_l):
        constraints_ += "\tx" + str(i) + "inreal: TYPE = { r: real | r>=-<TBD> AND r<=<TBD> }\n"

    return constraints_


# Read input arguments
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='torch2pvs',
        description='Generate a skeleton PVS theory for a MLP torch model',
        epilog='Federico Rossi, 2023')

    parser.add_argument("model_path")
    parser.add_argument("-o", "--output_vars", default=1, required=False, type=int)
    parser.add_argument("-p", "--pvs_output", default=None, required=False)
    parser.add_argument("-n", "--name", default="mlp", required=False)
    parser.add_argument("-s", "--sym", required=False, action='store_true')
    args = parser.parse_args()

    model = torch.load(args.model_path, map_location=torch.device('cpu'))

    input_vars = next(model.parameters()).size()[1]

    lines = gen_matrix_declarations(model, args.sym)
    sequence = gen_network_operation_sequence(model)

    theorem = gen_theorem("network_bounds", input_vars)

    constraints = gen_constraint_expressions(input_vars)

    wb_axioms = []

    if args.sym is True:
        wb_axioms = gen_axioms_from_sym_vars()

    # Emit PVS

    pvs_buffer = ""
    pvs_buffer += "%" + str(model).replace("\n", "\n%") + "\n"
    pvs_buffer += args.name + const_header

    if args.sym is True:
        pvs_buffer += "\n\t" + ",".join(sym_vars.keys()) + ": VAR real"
        pvs_buffer += "\n\t" + "\n\t".join(wb_axioms)

    pvs_buffer += "\n\t"
    for line in lines:
        pvs_buffer += line + "\n\t"

    pvs_buffer += "\n" + const_mat_act

    pvs_buffer += "\n" + constraints

    pvs_buffer += "\n\t" + const_network
    pvs_buffer += "\t\t\n\t\t" + "".join(sequence) + "\n\n"

    pvs_buffer += "\t" + theorem
    pvs_buffer += "\n\n" + const_trailer + args.name

    if args.pvs_output is not None:
        outfile = open(args.pvs_output, "w")
        outfile.write(pvs_buffer)
    else:
        print(pvs_buffer)


