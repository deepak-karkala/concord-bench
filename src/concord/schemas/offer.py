from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class EcommerceOffer(BaseModel):
    domain: Literal["ecommerce"] = "ecommerce"
    price: float = Field(ge=0, description="Total price in USD")
    quantity: int = Field(ge=1, description="Number of units")
    shipping_terms: str = Field(default="standard", description="Shipping arrangement")
    return_policy: str = Field(default="30-day", description="Return policy description")


class SaaSProcurementOffer(BaseModel):
    domain: Literal["saas_procurement"] = "saas_procurement"
    monthly_price: float = Field(ge=0, description="Monthly subscription price per seat")
    seats: int = Field(ge=1, description="Number of licensed seats")
    contract_length_months: int = Field(ge=1, description="Contract duration in months")
    sla_tier: str = Field(default="standard", description="Service level agreement tier")


class SettlementOffer(BaseModel):
    domain: Literal["settlement"] = "settlement"
    settlement_amount: float = Field(ge=0, description="Total settlement payment")
    payment_terms: str = Field(default="lump_sum", description="Payment structure")
    confidentiality_clause: bool = Field(default=False, description="Includes NDA clause")
    non_disparagement: bool = Field(default=False, description="Includes non-disparagement clause")


class EthicalBusinessOffer(BaseModel):
    domain: Literal["ethical_business"] = "ethical_business"
    price: float = Field(ge=0, description="Total price in USD")
    environmental_commitments: list[str] = Field(
        default_factory=list,
        description="Environmental commitments made by supplier",
    )
    labor_standards: list[str] = Field(
        default_factory=list,
        description="Labor standard commitments",
    )
    transparency_reports: bool = Field(
        default=False, description="Agreement to publish transparency reports"
    )


Offer = Annotated[
    Union[EcommerceOffer, SaaSProcurementOffer, SettlementOffer, EthicalBusinessOffer],
    Field(discriminator="domain"),
]
