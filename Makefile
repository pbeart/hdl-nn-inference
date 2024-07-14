.PHONY: all clean

# useful to keep generated SV files for inspection/debugging
.PRECIOUS: generated/%.sv

#--trace-depth 2 
VERILATOR_CMD := verilator --cc --exe --trace --build -j 0 \
-Wall --Mdir ./obj_dir -Wno-ENUMVALUE -Wno-WIDTHEXPAND -Wno-ALWCOMBORDER -Wno-UNOPTFLAT -Wno-UNUSED -y ip

csv_sources := $(wildcard test/*_tb_csv.cpp)
unit_sources := $(wildcard test/*_tb_unit.cpp)

all_tb_sources := $(csv_sources) $(unit_sources)
all_tb_plainnames := $(all_tb_sources:test/%=%)

all_tb_targets := $(all_tb_plainnames:%.cpp=build/%)

all: $(all_tb_targets)

# $(word 2, $^) is 2nd source file
./obj_dir/%_tb_generated: test/%_tb_generated.cpp generated/%.sv $(wildcard test/*.hh) $(wildcard ip/*)
	$(VERILATOR_CMD) $(word 2,$^) --exe $< -o "$*_tb_generated"

./obj_dir/%_tb_unit : test/%_tb_unit.cpp ip/%.sv $(wildcard test/*.hh)
	$(VERILATOR_CMD) $(word 2,$^) --exe $< -o "$*_tb_unit"

./obj_dir/%_tb_csv : test/%_tb_csv.cpp ip/%.sv $(wildcard test/*.hh)
	$(VERILATOR_CMD) $(word 2,$^) --exe $< -o "$*_tb_csv"

./build/%: ./obj_dir/%
	mkdir -p "build" && cp -f "$<" "$@"

clean:
	rm -rf build obj_dir generated
