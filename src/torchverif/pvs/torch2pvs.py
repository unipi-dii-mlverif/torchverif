import torch
import torch.nn as nn
import argparse



# Constant blocks

# Header for theory imports,
# add new imports here
const_header = ''': THEORY
    BEGIN 
'''

const_trailer = "END "
relu_strategy = '''
%|- network_bounds: PROOF
%|- (skeep)
%|- (expand "net0")
%|- (expand "relu")
%|- (repeat (lift-if))
%|- (try (prop) ( assert ) (skip))
%|- QED
'''

sym_vars = {}
# Generates the PVS function signature for a leaky relu with slope 'slope'
def get_leaky_relu_string(slope):
    return f'lrelu(x: real): real = IF x > 0 THEN x ELSE {slope}*x ENDIF'


def gen_vector_matrix_product(v, mw, mb, activation):
    row_prod = []
    np_mb = mb.detach().numpy()
    for r, row in enumerate(mw):
        col_prod = []
        b = np_mb[r]
        for c, col in enumerate(row):
            w = col.item()
            col_prod.append("(" + v[c] + ") * (" + str(w) + ")")
        row_prod.append(activation + "(" + '+'.join(col_prod) + " + " + str(b) + ")")
    formatted_row_prod = "(: " + ",".join(row_prod) + " :)"

    return row_prod, formatted_row_prod


# Generates the succession of function calls to perform network inference
# for the model 'model'
def gen_network_operation_sequence(model_, var_names):
    network_operations = []
    last_output = []
    formatted_outputs = []
    for i_, layer in enumerate(model_):
        if isinstance(layer, nn.Linear):
            if i_ == 0:
                network_operations.append("input")
                last_output, f = gen_vector_matrix_product(var_names, layer.weight, layer.bias, "relu")
                formatted_outputs.append(f)
            else:
                network_operations.append("*linear" + str(i_) + "+linear_bias" + str(i_))
                last_output, f = gen_vector_matrix_product(last_output, layer.weight, layer.bias, "")
                formatted_outputs.append(f)

    return network_operations, formatted_outputs, last_output


# Generates a placeholder theorem
def gen_theorem(name_, input_vars_):
    name_ += ": THEOREM\n"
    name_ += "\t\tFORALL ("

    x_input_vars = []
    x_input_names = []
    for i_ in range(input_vars_):
        x_input_names.append("x" + str(i_))
        x_input_vars.append("x" + str(i_) + ": x" + str(i_) + "t")

    name_ += ','.join(x_input_vars) + "):\n"
    name_ += "\t\t\t net0( " + ','.join(x_input_names) + "  ) >= 1"

    return name_, x_input_vars, x_input_names


def gen_theorem_eval(name_):
    name_ += ": THEOREM\n"
    name_ += "\t\t net0(0,0,0,0) = 0"
    return name_


# Generates placeholder constraints on input variables 'input_vars'
def gen_constraint_expressions(input_vars_l):
    constraints_ = ""
    for i_ in range(input_vars_l):
        constraints_ += "\tx" + str(i_) + "t: TYPE = { r: real | r>=-1 AND r<=1 }\n"
    return constraints_


def emit_pvs_from_pth(pth_path, name="nn_theory", output_path=None):
    model = torch.load(pth_path, map_location=torch.device('cpu'))
    input_vars = next(model.parameters()).size()[1]
    theorem, x_vars, x_names = gen_theorem("network_bounds", input_vars)
    sequence, _, lo = gen_network_operation_sequence(model, x_names)
    constraints = gen_constraint_expressions(input_vars)

    # PVS buffer gen
    pvs_buffer = ""
    pvs_buffer += "%" + str(model).replace("\n", "\n%") + "\n"
    pvs_buffer += name + const_header
    pvs_buffer += "\n\t"
    pvs_buffer += "\n" + constraints
    pvs_buffer += "\n\t"
    pvs_buffer += "relu(x:real):real= IF x > 0 THEN x ELSE 0 ENDIF"
    for i, net in enumerate(lo):
        pvs_buffer += "\n\t" + "net" + str(i) + "(" + ','.join(x_vars) + "): real = "
        pvs_buffer += "\t\t\n\t\t" + net + "\n\n"
    pvs_buffer += "\t" + theorem
    pvs_buffer += "\n" + relu_strategy
    pvs_buffer += "\n\n" + const_trailer + name
    pvs_buffer += "\n\t"

    if output_path is not None:
        outfile = open(output_path, "w")
        outfile.write(pvs_buffer)
    else:
        print(pvs_buffer)


# Read input arguments
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='torch2pvs',
        description='Generate a skeleton PVS theory for a MLP torch model',
        epilog='Federico Rossi, 2024')

    parser.add_argument("model_path")
    parser.add_argument("-o", "--outfile", default=None, required=False)
    args = parser.parse_args()
    emit_pvs_from_pth(args.model_path, args.outfile)
