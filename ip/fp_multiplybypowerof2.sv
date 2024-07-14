module fp_multiplybypowerof2(argumenta, out);
	parameter floatsize = 16;
	parameter negate = 0;
	parameter exponentsize = 5;
	//localparam exponent_bias = 15;
	localparam significandsize = floatsize-exponentsize-1;
	
	parameter power = 4;

	input [floatsize-1:0] argumenta;
	output reg [floatsize-1:0] out;

	initial begin
		//$display("Floatsize ", 16, " exponentsize ", exponentsize, " significandsize ", significandsize);
	end
	
	wire sign_a = argumenta[floatsize-1];

	wire [exponentsize-1:0] exponent_a = argumenta[floatsize-2:floatsize-1-exponentsize];

	wire [significandsize-1:0] significand_a = argumenta[significandsize-1:0];


	wire [exponentsize-1:0] exponent_a_subtracted = exponent_a + exponentsize'(power);
	
	wire [floatsize-1:0] affected = {negate ? ~sign_a : sign_a, exponent_a_subtracted, significand_a};
	
	always_comb begin
		if (power > 0) begin
			if (exponent_a_subtracted < exponent_a) begin
				out = '{default: '0};
			end else begin
				out = affected;
			end
		end else begin
			if (exponent_a_subtracted > exponent_a) begin
				out = '{default: '0};
			end else begin
				out = affected;
			end
		end
	end
		
endmodule
