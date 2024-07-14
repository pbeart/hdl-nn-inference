module fp_sum (argument_array, out);
	parameter floatsize = 16;
	parameter exponentsize = 5;
	parameter inputcount = 4;
	//localparam exponent_bias = 15;
	
	localparam significandsize = floatsize-exponentsize-1;

	input [floatsize-1:0] argument_array [inputcount];
	output reg [floatsize-1:0] out;
	
	reg [exponentsize-1:0] max_exponent;
	
	
	// +2 for the implied leading 1 and the significand sign, 
	// clog2(inputcount) for the carries
	reg [significandsize-1 + 3 + $clog2(inputcount):0] significand_sum;
	
	reg [exponentsize-1:0] this_exponent;
	
	reg [exponentsize-1:0] this_old_exponent;
	
	reg [significandsize-1:0] this_significand;
	reg this_is_zero;
	
	reg [significandsize-1 + 2:0] this_sext_significand;
	
	reg [significandsize-1 + 2:0] this_shifted_significand;
	
	reg significand_result_sign;

	wire [significandsize-1:0] significand_correct = significand_sum[$size(significand_sum)-1 - 2:$size(significand_sum)-1 - significandsize + 1 - 2];

	
	reg signed [$clog2($size(significand_sum)) - 1:0] significand_carry;
	
	reg this_sign;
	
	always_comb begin
		max_exponent = argument_array[0][floatsize-2:floatsize-1-exponentsize];
		
		
		// Determine the maximum exponent
		for (int i=1; i<inputcount; i++) begin
			this_exponent = argument_array[i][floatsize-2:floatsize-1-exponentsize];
			max_exponent = (this_exponent > max_exponent) ? this_exponent : max_exponent;
		end

		this_is_zero = argument_array[0] == '0;
		
		this_old_exponent = argument_array[0][floatsize-2:floatsize-1-exponentsize];

		this_significand = argument_array[0][significandsize-1:0];
		this_sign = argument_array[0][floatsize-1];
		this_sext_significand = {1'b0, 1'b1, this_significand};
		//$display("First element sext significand ", this_sext_significand);
		this_shifted_significand = this_sext_significand >> (max_exponent - this_old_exponent);
		//$display("First element shifted significand ", this_shifted_significand);
		if (this_sign == 1'b0) begin
			significand_sum = this_is_zero ? '0 : (this_shifted_significand);
		end else begin
			significand_sum = this_is_zero ? '0 : -(this_shifted_significand);
		end
		//$display("-- begin loop--");
		for (int i=1; i<inputcount; i++) begin
			//$display(" - Summing element @", i, ": ", argument_array[i], " - ");
			this_is_zero = argument_array[i] == '0;
			this_significand = argument_array[i][significandsize-1:0];
			this_sign = argument_array[i][floatsize-1];
			this_old_exponent = argument_array[i][floatsize-2:floatsize-1-exponentsize];
			//$display("This significand ", this_significand, " size ", $size(this_significand));
			//$display("This current exponent " , this_old_exponent);
			this_sext_significand = {2'b01, this_significand};
			//$display("Sext significand ", this_sext_significand, " size ", $size(this_sext_significand));
			this_shifted_significand = this_is_zero ? '0 : this_sext_significand >> (max_exponent - this_old_exponent);
			//$display("This corrected significand ", this_shifted_significand);
			if (this_sign == 1'b0) begin // Positive
				significand_sum = significand_sum + this_shifted_significand;
				//$display("Adding and getting", significand_sum);
			end else begin
				significand_sum = significand_sum - this_shifted_significand;
				//$display("substract");
			end
		end
		//$display("-- end loop --");
		// Was result negative? If so, keep track of this, and flip bits to make it positive
		//$display("Pre messing about with it significand sum", significand_sum);
		if (significand_sum[$size(significand_sum)-1] == 1'b1) begin
			significand_result_sign = 1'b1;
			significand_sum = ~significand_sum;
		end else begin
			significand_result_sign = 1'b0;
		end
		significand_carry = 4'($clog2(inputcount)) + 4'b1; // start with maximum carry, then count down for every bit away from max the overflow is
		
		for (int i=0; i<$size(significand_sum); i++) begin
			if (significand_sum[$size(significand_sum) - 2] == 1'b1) begin// - i] == 1'b1) begin
				break;
			end
			significand_carry = significand_carry - 1;
			significand_sum = significand_sum << 1;
		end
		//$display("Final shifted whole sig sum ", significand_sum, " corrected is ", significand_correct);
		//$display("sigca", significand_carry);
		//$display("Final significand old ", significand_sum[significandsize-1:0]);
		//$display("Significand new ", significand_correct);
		out = significand_sum == '0 ? '0 : {significand_result_sign, max_exponent + significand_carry/* - exponentsize'($size(significand_sum) - $size(this_significand))*/, significand_correct};
		
	end
endmodule
