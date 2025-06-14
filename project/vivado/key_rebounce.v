`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2025/06/14 11:37:17
// Design Name: 
// Module Name: key_rebounce
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
module key_rebounce(
    input I_clk,
    input I_rst_n,
    input I_key_in,
    output reg o_key_out
);

    reg [19:0] cnt;
    reg key_reg;

    wire change = I_key_in ^ key_reg;

    always @(posedge I_clk or negedge I_rst_n) begin
        if (!I_rst_n) begin
            cnt <= 0;
            key_reg <= 0;
        end else begin
            key_reg <= I_key_in;
            if (change)
                cnt <= 0;
            else if (cnt < 20'd999_999)
                cnt <= cnt + 1;
        end
    end

    always @(posedge I_clk or negedge I_rst_n) begin
        if (!I_rst_n)
            o_key_out <= 0;
        else if (cnt == 20'd999_999)
            o_key_out <= key_reg;
    end

endmodule
