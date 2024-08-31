import itertools
import torch

def estimate_lipschitz_constant(model, inputs, n_estimates=None):
	
	inputs = inputs.requires_grad_(True)

	outputs = model(inputs)
	
	loss = outputs.mean()
	
	model.zero_grad()
	
	loss.backward(retain_graph=True)
	
	n_nodes = inputs.shape[0]
	lipschitz_constant = -float('inf')
	
	combinations = list(itertools.combinations(range(0, n_nodes), 2))
	num_combinations = (n_nodes * (n_nodes - 1)) // 2
	
	if n_estimates:
		print(f'Checking {n_estimates} combinations.')
	else:
		print(f'Checking {num_combinations} combinations.')

	combination_checkpoints = set([(num_combinations * i) // 10 for i in range(1, 11)])
	
	for index, (num1, num2) in enumerate(combinations):
		x_diff = inputs[num1] - inputs[num2]
		x_diff_norm = x_diff.norm(2)
		
		if x_diff_norm == 0:
			continue
		
		if n_estimates and index == n_estimates:
			return lipschitz_constant.item()

		grad_num1 = torch.autograd.grad(outputs=outputs[num1], inputs=inputs, 
							grad_outputs=torch.ones_like(outputs[num1]),
							retain_graph=True, create_graph=False)[0][num1]

		grad_num2 = torch.autograd.grad(outputs=outputs[num2], inputs=inputs, 
							grad_outputs=torch.ones_like(outputs[num2]),
							retain_graph=True, create_graph=False)[0][num2]

		grad_diff = grad_num1 - grad_num2
		grad_diff_norm = grad_diff.norm(2)

		new_lipschitz_constant = grad_diff_norm / x_diff_norm
		lipschitz_constant = max(lipschitz_constant, new_lipschitz_constant)

		if n_estimates and index + 1 in combination_checkpoints:
			print(f'Checked {index + 1}/{n_estimates} Combinations')
		elif index + 1 in combination_checkpoints:
			print(f'Checked {index + 1}/{num_combinations} Combinations')

	return lipschitz_constant.item()