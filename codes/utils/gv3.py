import gvar as gv
import math

class gv2:
    def __init__(self, value_str):
        """
        Initialize a data object with one, two, or three uncertainties.
        The input format can be:
        - 'mean(uncertainty1)': Single uncertainty
        - 'mean(uncertainty1)(uncertainty2)': Two uncertainties
        - 'mean(uncertainty1)(uncertainty2)(uncertainty3)': Three uncertainties
        """
        try:
            # Split the input into the mean and uncertainty parts
            parts = value_str.split('(')
            self.mean = parts[0].strip()

            # Parse the uncertainties
            errors = [str(p.strip(')')) for p in parts[1:]]

            if len(errors) == 1:
                # Single uncertainty
                # self.error1 = errors[0]
                # print(self.mean,errors[0])
                self.error1 = gv.gvar(f'{self.mean}({errors[0]})').sdev
                self.error2 = 0.0
                self.error3 = 0.0
            elif len(errors) == 2:
                # Two uncertainties
                # self.error1, self.error2 = errors
                self.error1, self.error2 = gv.gvar(f'{self.mean}({errors[0]})').sdev,gv.gvar(f'{self.mean}({errors[1]})').sdev
                self.error3 = 0.0
            elif len(errors) >= 3:
                # Three uncertainties (ignore extra ones)
                self.error1, self.error2, self.error3 =  gv.gvar(f'{self.mean}({errors[0]})').sdev,gv.gvar(f'{self.mean}({errors[1]})').sdev,gv.gvar(f'{self.mean}({errors[2]})').sdev

            else:
                raise ValueError("Input must contain at least one uncertainty.")

            # Calculate total uncertainty (root mean square)
            self.sdev = math.sqrt(self.error1**2 + self.error2**2 + self.error3**2)

            # Create a gvar object for further computations
            self.gvar = gv.gvar(float(self.mean), self.sdev)
        except Exception as e:
            raise ValueError(f"Parsing error: {e}")

    def __repr__(self):
        # Represent the object with its mean and uncertainties
        return f"{self.mean}({self.error1:.2f})({self.error2:.2f})({self.error3:.2f})"

# Example usage
examples = ['0.5(1)', '0.5(1)(1)', '0.5(1)(1)']

for ex in examples:
    print(f"Input: {ex}")
    try:
        a = gv2(ex)
        print(f"Mean: {a.mean}")
        print(f"Uncertainty 1: {a.error1}")
        print(f"Uncertainty 2: {a.error2}")
        print(f"Uncertainty 3: {a.error3}")
        print(f"Total Uncertainty (RMS): {a.sdev}")
        print(f"Corresponding gvar object: {a.gvar}")
    except ValueError as e:
        print(f"Error: {e}")
    print()
