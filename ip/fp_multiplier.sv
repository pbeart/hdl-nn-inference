module fp_multiplier(argumenta, out);
	parameter floatsize = 32;
	parameter [floatsize-1:0] multiplicand = 32'h42200000;
	parameter exponentsize = 8;

	input [floatsize-1:0] argumenta;
	output [floatsize-1:0] out;

	localparam [exponentsize-1:0] exponent_bias = 127;
	localparam significandsize = floatsize-exponentsize-1;


	wire sign_b = multiplicand[floatsize-1];

	wire [exponentsize-1:0] exponent_b = multiplicand[floatsize-2:floatsize-1-exponentsize];


	wire [significandsize-1:0] significand_b = multiplicand[significandsize-1:0];

	wire [significandsize:0] significand_b_sext = {1'b1, significand_b};


	wire sign_a = argumenta[floatsize-1];

	wire [exponentsize-1:0] exponent_a = argumenta[floatsize-2:floatsize-1-exponentsize];

	wire [significandsize-1:0] significand_a = argumenta[significandsize-1:0];

	wire [significandsize:0] significand_a_sext = {1'b1, significand_a};



	wire [exponentsize-1:0] exponent_out_unnorm = exponent_a + exponent_b - exponent_bias;

	wire overflow = (exponent_a > 0) && (exponent_b > 0) && (exponent_out_unnorm<0);
	wire underflow = (exponent_a < 0) && (exponent_b < 0) && (exponent_out_unnorm>0);


	wire [2*significandsize + 2:0] significand_multiplied_unnorm = significand_a_sext * significand_b_sext;

	reg [2*significandsize + 2:0] significand_shifted;

	reg [5:0] significand_first_1_index;

	always_comb begin
		significand_shifted = significand_multiplied_unnorm;
		significand_first_1_index = 0;
		for (integer i = 0; i < significandsize; i++) begin
			significand_shifted = significand_shifted << 1;
			if (significand_shifted[2*significandsize + 2] == 1) begin
				 significand_first_1_index = i - 1'b1;
				 break;
			end
		end
	end



	wire [significandsize-1:0] significand_out = significand_shifted[2*significandsize + 1:significandsize + 2];

	wire [exponentsize-1:0] exponent_out = exponent_out_unnorm - significand_first_1_index;

	wire sign_out = sign_a ^ sign_b;

	assign out = (argumenta == 0) ? '{default: '0} : {sign_out, exponent_out, significand_out};
		
endmodule
