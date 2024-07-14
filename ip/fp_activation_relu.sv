module fp_activation_relu(argumenta, out);
	parameter floatsize = 32;
	parameter exponentsize = 8;

	if (exponentsize != 0); // prevent unused warning. ugly.

	input [floatsize-1:0] argumenta;
	output reg [floatsize-1:0] out;

	wire sign_a = argumenta[floatsize-1];

	always_comb begin
		if (sign_a == 0) out = argumenta;
		else out = '{default: '0};
	end
		
endmodule
