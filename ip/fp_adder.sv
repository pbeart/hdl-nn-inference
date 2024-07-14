module fp_adder(argumenta, argumentb, out);
	parameter floatsize = 16;
	parameter exponentsize = 5;
	localparam [exponentsize-1:0] exponent_bias = 15;
	localparam significandsize = floatsize-exponentsize-1;
	
	/*parameter floatsize = 32;
	parameter exponentsize = 8;
	localparam [exponentsize-1:0] exponent_bias = 127;
	localparam significandsize = floatsize-exponentsize-1;*/
	
	
	input [floatsize-1:0] argumenta;
	input [floatsize-1:0] argumentb;
	output reg [floatsize-1:0] out;
	
	
	wire sign_b = argumentb[floatsize-1];
	wire [exponentsize-1:0] exponent_b = argumentb[floatsize-2:floatsize-1-exponentsize];
	wire [significandsize-1:0] significand_b = argumentb[significandsize-1:0];
	


	wire sign_a = argumenta[floatsize-1];
	wire [exponentsize-1:0] exponent_a = argumenta[floatsize-2:floatsize-1-exponentsize];
	wire [significandsize-1:0] significand_a = argumenta[significandsize-1:0];
	
	
	wire argsorter = {exponent_a, significand_a} >= {exponent_b, significand_b};
	
	wire sign_big = argsorter ? sign_a : sign_b;
	wire [exponentsize-1:0] exponent_big = argsorter ? exponent_a : exponent_b;
	wire [significandsize-1:0] significand_big = argsorter ? significand_a : significand_b;
	wire [significandsize:0] significand_big_sext = {1'b1, significand_big};
	
	wire sign_small = argsorter ? sign_b : sign_a;
	wire [exponentsize-1:0] exponent_small_unadjusted = argsorter ? exponent_b : exponent_a;
	wire [significandsize-1:0] significand_small_unadjusted = argsorter ? significand_b : significand_a;
	wire [significandsize:0] significand_small_sext_unadjusted = {1'b1, significand_small_unadjusted};


	
	// Shift smaller up to bigger
	wire [significandsize:0] significand_small_sext_adjusted = (significand_small_sext_unadjusted >> (exponent_big - exponent_small_unadjusted));//+ (exponent_big - exponent_small_unadjusted) * significand_small_sext_unadjusted[exponent_big - exponent_small_unadjusted - 1];
	
	
	wire [significandsize+1:0] big_sext_operand = {1'b0, significand_big_sext};
	
	wire [significandsize+1:0] small_sext_operand = {1'b0, significand_small_sext_adjusted};
	
	wire [significandsize+1:0] big_sext_operand_signed = sign_big ? (-big_sext_operand) : big_sext_operand;
	wire [significandsize+1:0] small_sext_operand_signed = sign_small ? (-small_sext_operand) : small_sext_operand;
	
	
	wire [significandsize+2:0] significands_added = big_sext_operand_signed + small_sext_operand_signed;
	
	
	
	wire significand_sign = significands_added[significandsize+2];
	
	//wire significand_carry = significands_added[significandsize+1] ^ significand_sign; // because carry will be 1 if no overflow and negative out!
	
	wire [significandsize+1:0] significand_for_shifting = (significand_sign ? (~significands_added[significandsize+1:0]) : significands_added[significandsize+1:0]);
	
	reg [significandsize+1:0] significand_shifted;

	// todo: resize this appropriately (clog2 of significandsize)
	reg [$clog2(significandsize+1) - 1:0] significand_first_1_index;

	always_comb begin
		significand_shifted = significand_for_shifting;
		significand_first_1_index = 0;
		for (integer i = 0; i < significandsize; i++) begin
			
			if (significand_shifted[significandsize+1] == 1) begin
				 significand_first_1_index = $size(significand_first_1_index)'(i);
				 break;
			end
			significand_shifted = significand_shifted << 1;
		end
	end
	
	wire [significandsize-1:0] significands_added_shifted = significandsize'(significand_shifted >> 1);// >> significand_first_1_index;
	
	
	wire [significandsize-1:0] significand_out = significands_added_shifted + significand_shifted[0]; // add the rounded bit

	
	
	
	// todo: subtract don't add if carry and significand_sign
	always_comb begin
		if (significands_added == 0) begin
			out = 0;
		end else begin
			out = (argumenta == 0) ? argumentb : ((argumentb == 0) ? argumenta : {significand_sign, exponent_big - significand_first_1_index + 1'b1, significand_out});
		end
	end
	
		
endmodule
